"""Startup is watchable, and a failed startup does not pretend otherwise.

Warmup used to run inside the lifespan, so uvicorn accepted no connections until the feed was
parsed and two upstreams answered — a browser saying "cannot connect" for tens of seconds on a
first launch. It now runs behind the server, and `/` serves a board reporting real progress
until it is genuinely usable.
"""

from __future__ import annotations

from tripps.api.app import BootProgress


def test_a_fresh_boot_reports_nothing_done() -> None:
    p = BootProgress().payload()
    assert p["ready"] is False
    assert p["failed"] is False
    assert p["progress"] == 0.0
    assert [s["state"] for s in p["steps"]] == ["pending", "pending", "pending"]


def test_steps_advance_and_carry_their_result() -> None:
    b = BootProgress()
    b.begin("feed")
    assert b.payload()["steps"][0]["state"] == "active"

    b.finish("feed", "1 298 stops · 5 743 departures")
    p = b.payload()
    assert p["steps"][0]["state"] == "done"
    assert p["steps"][0]["detail"] == "1 298 stops · 5 743 departures"
    assert p["progress"] == 1 / 3


def test_progress_reaches_one_only_when_every_step_is_done() -> None:
    b = BootProgress()
    for key in ("feed", "cars", "fares"):
        b.begin(key)
        b.finish(key, "ok")
    assert b.payload()["progress"] == 1.0


def test_a_failed_step_is_flagged_and_keeps_its_remedy() -> None:
    """The boot screen stays put on failure, so this text is what the person acts on."""
    b = BootProgress()
    b.begin("feed")
    b.fail("feed", "No timetable feed yet — run `tripps fetch-gtfs`.")

    p = b.payload()
    assert p["failed"] is True
    assert p["steps"][0]["state"] == "failed"
    assert "tripps fetch-gtfs" in p["steps"][0]["detail"]
    # A failure must not count toward completion.
    assert p["progress"] == 0.0


def test_a_failure_does_not_block_the_later_steps() -> None:
    """A missing feed should not also hide the fact that cars and fares are fine."""
    b = BootProgress()
    b.fail("feed", "no feed")
    b.finish("cars", "87 free cars in Sweden")
    b.finish("fares", "live prices ready")

    p = b.payload()
    assert p["failed"] is True
    assert [s["state"] for s in p["steps"]] == ["failed", "done", "done"]


def test_elapsed_is_reported_for_the_clock() -> None:
    p = BootProgress().payload()
    assert isinstance(p["elapsed"], float)
    assert p["elapsed"] >= 0.0
