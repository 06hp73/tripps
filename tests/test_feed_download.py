"""An interrupted feed download must not brick the folder.

Found by killing a `./run.sh` mid-download: 65 MB is long enough to be interrupted, and the
old code streamed straight into `data/sweden.zip`. Every later run then saw a feed present,
skipped the download, and died on `BadZipFile` five frames deep in the parser — with nothing
telling the person to delete the file. The fix is to download beside the target and rename
only once the bytes are whole, plus an actionable message if a bad feed is already there.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
import pytest
import respx

from tripps.cli import _fetch_gtfs
from tripps.config import Settings, reset_settings_cache
from tripps.ingest.gtfs import load_timetable

FEED_URL = "https://api.resrobot.se/gtfs/sweden.zip"


def _real_zip() -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("agency.txt", "agency_id,agency_name\n1,Test\n")
    return buf.getvalue()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("TRIPPS_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    yield tmp_path
    reset_settings_cache()


@respx.mock
async def test_interrupted_download_leaves_no_feed_behind(data_dir: Path) -> None:
    """A mid-stream failure must leave the target absent, not truncated."""
    respx.get(FEED_URL).mock(
        side_effect=httpx.ReadError("connection dropped at 40 MB")
    )
    with pytest.raises(httpx.ReadError):
        await _fetch_gtfs()

    target = data_dir / "sweden.zip"
    assert not target.exists(), "a failed download must not leave a file that looks usable"
    assert not (data_dir / "sweden.zip.part").exists(), "the partial must be cleaned up too"


@respx.mock
async def test_a_non_zip_response_is_refused(data_dir: Path) -> None:
    """An error page served with HTTP 200 is a failed download, whatever the status said."""
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"<html>upstream is down</html>")
    )
    assert await _fetch_gtfs() == 1
    assert not (data_dir / "sweden.zip").exists()


@respx.mock
async def test_a_complete_download_lands_atomically(data_dir: Path) -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=_real_zip()))
    assert await _fetch_gtfs() == 0

    target = data_dir / "sweden.zip"
    assert zipfile.is_zipfile(target)
    assert not (data_dir / "sweden.zip.part").exists()


def test_a_corrupt_feed_says_what_to_do(tmp_path: Path) -> None:
    """Whatever the cause, the message names the file and the command that fixes it."""
    bad = tmp_path / "sweden.zip"
    bad.write_bytes(b"PK\x03\x04truncated...")

    from datetime import date

    with pytest.raises(ValueError) as exc:
        load_timetable(bad, date(2026, 8, 9))
    message = str(exc.value)
    assert str(bad) in message
    assert "tripps fetch-gtfs" in message


def test_the_agency_stop_path_refuses_a_corrupt_feed_too(tmp_path: Path) -> None:
    """extract_agency_stops opens the feed before the parser does, so it fails first.

    Fixing only load_timetable left the real traceback untouched: the search path reached
    _compute_agency_stops first and raised BadZipFile from there. Both go through open_feed.
    """
    from tripps.ingest.gtfs import _compute_agency_stops

    bad = tmp_path / "sweden.zip"
    bad.write_bytes(b"PK\x03\x04truncated...")

    with pytest.raises(ValueError, match="tripps fetch-gtfs"):
        _compute_agency_stops(bad)


def test_the_partial_path_follows_a_relocated_data_dir(tmp_path: Path) -> None:
    """The .part file must not be written back into the repo when the data dir moves."""
    s = Settings(data_dir=tmp_path)
    partial = s.gtfs_zip_path.with_suffix(s.gtfs_zip_path.suffix + ".part")
    assert partial.parent == tmp_path
