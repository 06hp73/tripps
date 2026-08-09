"""A relocated data dir must take the feed and the database with it.

Both paths default to `DATA_DIR / ...`, evaluated when the class is defined. Without the
validator, `TRIPPS_DATA_DIR=/somewhere/else` moved the directory and nothing inside it:
`ensure_dirs()` created the new directory while every read still went to the repo checkout.
Docker exposes exactly this knob, so the split state was reachable from a compose file.
"""

from __future__ import annotations

from pathlib import Path

import tripps.config as config_module
from tripps.config import DATA_DIR, Settings, _default_data_dir


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


# --- where the default itself comes from ------------------------------------------------
#
# A packaged .app has no checkout around it, and must not write into its own bundle: it may
# sit in /Applications, be quarantined, or be replaced wholesale on update. So the default
# has to depend on whether a source tree is present, not on where the package happens to live.


def test_explicit_env_beats_everything(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRIPPS_DATA_DIR", str(tmp_path / "chosen"))
    assert _default_data_dir() == tmp_path / "chosen"


def test_a_source_checkout_keeps_its_data_beside_it(monkeypatch) -> None:
    """This repo has a pyproject.toml two levels up, so nothing moves for a developer."""
    monkeypatch.delenv("TRIPPS_DATA_DIR", raising=False)
    assert _default_data_dir() == config_module.PROJECT_ROOT / "data"


def test_a_packaged_app_writes_to_the_user_data_dir(tmp_path: Path, monkeypatch) -> None:
    """No pyproject.toml above the package means it was installed, not checked out."""
    monkeypatch.delenv("TRIPPS_DATA_DIR", raising=False)
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path / "no-checkout-here")

    got = _default_data_dir()
    assert got.name == "tripps"
    assert tmp_path not in got.parents, "must not land inside the bundle"
    assert got.is_absolute()
    assert str(Path.home()) in str(got)
