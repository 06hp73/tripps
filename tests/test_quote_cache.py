"""Quote-cache lifecycle: estimates age like everything else, purges evict, blips serve stale.

The freshness contract: no cached number is ever presented as newer than it is. ESTIMATED used
to be exempt from TTL entirely ("fare tables do not go stale"), which froze FlixBus's dynamic
day-cheapest fallback forever - last month's price shown as today's.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tripps.config import PricingBudget
from tripps.db import Database
from tripps.models import Leg, PriceConfidence, Quote, Stop, TransportMode
from tripps.pricing.base import CallBudget
from tripps.pricing.orchestrator import PricingOrchestrator, _Context

A = Stop(id="A", name="A", lat=59.0, lon=18.0)
B = Stop(id="B", name="B", lat=57.7, lon=12.0)

FETCHED = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


def _leg() -> Leg:
    return Leg(
        from_stop=A, to_stop=B, mode=TransportMode.BUS, operator="FlixBus",
        departure=datetime(2026, 7, 24, 7, 30), arrival=datetime(2026, 7, 24, 13, 45),
        service_ref="f1",
    )


def _quote(confidence: PriceConfidence) -> Quote:
    return Quote(source="flixbus", amount_ore=41_900, confidence=confidence, fetched_at=FETCHED)


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "q.sqlite3")
    yield database
    database.close()


# --- ESTIMATED ages past its TTL -------------------------------------------------------------


def test_estimated_within_ttl_is_served_as_estimated(db):
    db.put_quote("flixbus", _leg(), _quote(PriceConfidence.ESTIMATED), ttl_seconds=1800)
    got = db.get_quote("flixbus", _leg(), now=FETCHED + timedelta(seconds=900))
    assert got is not None and got.confidence is PriceConfidence.ESTIMATED


def test_estimated_past_ttl_ages_to_stale(db):
    """The dynamic fallback fare is yield-managed; past TTL it must be labelled stale, which
    also makes the orchestrator re-call the adapter instead of short-circuiting forever."""
    db.put_quote("flixbus", _leg(), _quote(PriceConfidence.ESTIMATED), ttl_seconds=1800)
    got = db.get_quote("flixbus", _leg(), now=FETCHED + timedelta(days=30))
    assert got is not None and got.confidence is PriceConfidence.STALE


def test_exact_within_ttl_still_reads_cached_and_past_ttl_stale(db):
    db.put_quote("flixbus", _leg(), _quote(PriceConfidence.EXACT), ttl_seconds=1800)
    fresh = db.get_quote("flixbus", _leg(), now=FETCHED + timedelta(seconds=60))
    aged = db.get_quote("flixbus", _leg(), now=FETCHED + timedelta(hours=2))
    assert fresh.confidence is PriceConfidence.CACHED
    assert aged.confidence is PriceConfidence.STALE


# --- purge evicts dead rows -------------------------------------------------------------------


def test_purge_quotes_evicts_old_rows_and_keeps_recent(db):
    old = _quote(PriceConfidence.EXACT).model_copy(
        update={"fetched_at": datetime.now(UTC) - timedelta(days=10)}
    )
    db.put_quote("flixbus", _leg(), old, ttl_seconds=1800)
    recent_leg = _leg().model_copy(update={"service_ref": "f2"})
    db.put_quote("flixbus", recent_leg, _quote(PriceConfidence.EXACT).model_copy(
        update={"fetched_at": datetime.now(UTC)}
    ), ttl_seconds=1800)

    assert db.purge_quotes(timedelta(days=3)) == 1
    assert db.get_quote("flixbus", _leg()) is None
    assert db.get_quote("flixbus", recent_leg) is not None


# --- a transient adapter blip serves the stale cache, not a gap -------------------------------


class ExplodingAdapter:
    name = "flixbus"
    provides_price = True

    def supports(self, leg: Leg) -> bool:
        return True

    async def quote_leg(self, leg: Leg) -> Quote:
        raise RuntimeError("upstream blip")

    async def health(self):
        from tripps.interfaces import HealthState, SourceHealth

        return SourceHealth(self.name, HealthState.OK)

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_adapter_exception_serves_priced_stale_cache(db):
    """Same principle as the BudgetExceeded path: a stale number beats no number. Without the
    fallback, a transient blip unpriced the leg and require_priced hid the itinerary."""
    db.put_quote("flixbus", _leg(), _quote(PriceConfidence.EXACT), ttl_seconds=1800)
    orch = PricingOrchestrator(
        [ExplodingAdapter()], db, budget=PricingBudget(min_interval_seconds=0.0)
    )
    ctx = _Context(call_budget=CallBudget.from_settings(PricingBudget()))
    got = await orch._quote_leg(_leg(), ctx)
    assert got.is_priced
    assert got.confidence is PriceConfidence.STALE
    assert got.amount_ore == 41_900


@pytest.mark.asyncio
async def test_adapter_exception_without_cache_is_unavailable(db):
    orch = PricingOrchestrator(
        [ExplodingAdapter()], db, budget=PricingBudget(min_interval_seconds=0.0)
    )
    ctx = _Context(call_budget=CallBudget.from_settings(PricingBudget()))
    got = await orch._quote_leg(_leg(), ctx)
    assert not got.is_priced
    assert got.confidence is PriceConfidence.UNAVAILABLE
