"""A relocated data dir must take the feed and the database with it.

Both paths default to `DATA_DIR / ...`, evaluated when the class is defined. Without the
validator, `TRIPPS_DATA_DIR=/somewhere/else` moved the directory and nothing inside it:
`ensure_dirs()` created the new directory while every read still went to the repo checkout.
Docker exposes exactly this knob, so the split state was reachable from a compose file.
"""

from __future__ import annotations

from pathlib import Path

from tripps.config import DATA_DIR, Settings


def test_data_dir_moves_feed_and_db(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path)
    assert s.db_path == tmp_path / "tripps.sqlite3"
    assert s.gtfs_zip_path == tmp_path / "sweden.zip"


def test_explicit_paths_win_over_the_data_dir(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere" / "custom.sqlite3"
    s = Settings(data_dir=tmp_path, db_path=elsewhere)
    assert s.db_path == elsewhere
    # …and the path that was not named still follows the directory.
    assert s.gtfs_zip_path == tmp_path / "sweden.zip"


def test_defaults_are_untouched_when_the_data_dir_is_not_set() -> None:
    s = Settings()
    assert s.data_dir == DATA_DIR
    assert s.db_path == DATA_DIR / "tripps.sqlite3"
    assert s.gtfs_zip_path == DATA_DIR / "sweden.zip"


def test_env_var_relocates_everything(tmp_path: Path, monkeypatch) -> None:
    """The reported shape: TRIPPS_DATA_DIR set, nothing else."""
    monkeypatch.setenv("TRIPPS_DATA_DIR", str(tmp_path))
    s = Settings()
    assert s.data_dir == tmp_path
    assert s.gtfs_zip_path == tmp_path / "sweden.zip"
    assert s.db_path == tmp_path / "tripps.sqlite3"
