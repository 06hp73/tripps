"""Canary framework: result handling, exception-to-DOWN, persistence, probe-date choice.

The probes themselves hit live endpoints and are exercised by `tripps canary`, not here;
these tests pin the framework that must never itself raise.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tripps.db import Database
from tripps.interfaces import HealthState
from tripps.monitoring import CanaryResult, _probe_date, _run, persist_canaries, run_canaries


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
