"""Canary framework: result handling, exception-to-DOWN, persistence, probe-date choice.

The probes themselves hit live endpoints and are exercised by `tripps canary`, not here;
these tests pin the framework that must never itself raise.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tripps.db import Database
from tripps.interfaces import HealthState
from tripps.monitoring import (
    CanaryResult,
    _probe_date,
    _run,
    is_stale,
    persist_canaries,
    run_canaries,
)


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "m.sqlite3")
    yield database
    database.close()


def test_probe_date_is_a_weekday_a_few_days_out():
    # A Friday: +3 days lands on Monday, still a weekday.
    d = _probe_date(date(2026, 7, 10))
    assert d > date(2026, 7, 10)
    assert d.weekday() < 5


def test_probe_date_skips_the_weekend():
    # +3 days from this Wednesday is Saturday; must nudge to Monday.
    d = _probe_date(date(2026, 7, 8))
    assert d.weekday() < 5


def test_result_ok_covers_ok_and_degraded():
    assert CanaryResult("x", HealthState.OK, "", 1).ok
    assert CanaryResult("x", HealthState.DEGRADED, "", 1).ok
    assert not CanaryResult("x", HealthState.DOWN, "", 1).ok


def test_line_renders_each_state():
    for state in (HealthState.OK, HealthState.DEGRADED, HealthState.DOWN):
        line = CanaryResult("flixbus", state, "detail", 42).line()
        assert "flixbus" in line and "42" in line


async def test_run_turns_success_into_a_result():
    async def probe():
        return HealthState.OK, "all good"

    result = await _run("thing", probe())
    assert result.name == "thing"
    assert result.state is HealthState.OK
    assert result.detail == "all good"
    assert result.latency_ms >= 0


async def test_run_turns_an_exception_into_down_never_raises():
    async def probe():
        raise RuntimeError("endpoint exploded")

    result = await _run("thing", probe())
    assert result.state is HealthState.DOWN
    assert "endpoint exploded" in result.detail
    assert "RuntimeError" in result.detail


def test_persist_records_each_canary_under_a_prefixed_key(db):
    persist_canaries(
        db,
        [
            CanaryResult("flixbus", HealthState.OK, "10 departures", 500),
            CanaryResult("sj", HealthState.DOWN, "key rotated", 1200),
        ],
    )
    health = db.get_health()
    assert health["canary:flixbus"] == "ok"
    assert health["canary:sj"] == "down"


def test_stale_covers_missing_and_old_but_not_fresh():
    """A state with no timestamp is not evidence about now, so it counts as stale."""
    assert is_stale(None, 24)
    assert is_stale(24.1, 24)
    assert not is_stale(23.9, 24)
    assert not is_stale(0.0, 24)


def test_canary_age_is_none_until_a_probe_is_recorded(db):
    assert db.oldest_canary_age_hours() is None


def test_canary_age_tracks_the_oldest_probe_not_the_newest(db):
    """A source left behind by a partial run must still trigger a refresh.

    Keyed on the newest row, one just-probed source would vouch for the whole set and the
    stale one would never be re-probed - the exact failure the catch-up exists to prevent.
    """
    db.set_health("canary:flixbus", "degraded", "no bookable priced departures")
    with db._write() as conn:
        conn.execute(
            "UPDATE adapter_health SET checked_at=? WHERE name='canary:flixbus'",
            ((datetime.now(UTC) - timedelta(days=19)).isoformat(),),
        )
    persist_canaries(db, [CanaryResult("sj", HealthState.OK, "15 departures", 900)])

    age = db.oldest_canary_age_hours()
    assert age is not None and age > 24
    assert is_stale(age, 24)


def test_canary_age_is_fresh_once_every_source_is_reprobed(db):
    persist_canaries(
        db,
        [
            CanaryResult("flixbus", HealthState.OK, "8 departures", 660),
            CanaryResult("sj", HealthState.OK, "15 departures", 900),
        ],
    )
    age = db.oldest_canary_age_hours()
    assert age is not None and age < 1
    assert not is_stale(age, 24)


def test_only_canary_rows_count_toward_the_age(db):
    """Live per-request health is written on every search; it must not mask stale canaries."""
    db.set_health("canary:flixbus", "ok", "8 departures")
    with db._write() as conn:
        conn.execute(
            "UPDATE adapter_health SET checked_at=? WHERE name='canary:flixbus'",
            ((datetime.now(UTC) - timedelta(days=19)).isoformat(),),
        )
    db.set_health("freerider-inventory", "ok", "29 offers")

    age = db.oldest_canary_age_hours()
    assert age is not None and age > 24


def test_health_rows_carry_the_timestamp_get_health_drops(db):
    persist_canaries(db, [CanaryResult("flixbus", HealthState.DEGRADED, "no fares", 500)])
    row = db.get_health_rows()["canary:flixbus"]
    assert row["state"] == "degraded"
    assert "no fares" in row["detail"]
    assert row["checked_at"]


def test_health_entry_flags_an_old_probe_and_keeps_a_fresh_one(db):
    """What /health and the status page report about a stored row: state plus how old it is."""
    from tripps.api.app import _health_entry

    persist_canaries(db, [CanaryResult("sj", HealthState.OK, "15 departures", 900)])
    fresh = _health_entry(db.get_health_rows()["canary:sj"], 24)
    assert fresh["state"] == "ok"
    assert not fresh["stale"]
    assert "ago" in fresh["age_label"]

    old = dict(db.get_health_rows()["canary:sj"])
    old["checked_at"] = (datetime.now(UTC) - timedelta(days=19)).isoformat()
    entry = _health_entry(old, 24)
    assert entry["stale"]
    assert entry["age_label"] == "19 d ago"


def test_health_entry_without_a_timestamp_is_stale():
    from tripps.api.app import _health_entry

    entry = _health_entry({"state": "ok", "detail": "", "checked_at": None}, 24)
    assert entry["stale"]
    assert entry["age_label"] == "age unknown"


async def test_run_canaries_returns_one_result_per_source(monkeypatch):
    """The runner must yield a result for every source even if a probe fails, so a single
    dead endpoint never hides the health of the others."""
    import tripps.monitoring as mon

    async def ok_probe(*a, **k):
        return HealthState.OK, "fine"

    async def bad_probe(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(mon, "_probe_gtfs", ok_probe)
    monkeypatch.setattr(mon, "_probe_flixbus", ok_probe)
    monkeypatch.setattr(mon, "_probe_sj", bad_probe)
    monkeypatch.setattr(mon, "_probe_tora", ok_probe)
    monkeypatch.setattr(mon, "_probe_freerider", ok_probe)

    results = await run_canaries(day=date(2026, 7, 13))
    by_name = {r.name: r for r in results}
    assert set(by_name) == {"gtfs-feed", "flixbus", "sj", "tora", "freerider"}
    assert by_name["sj"].state is HealthState.DOWN
    assert by_name["flixbus"].state is HealthState.OK
