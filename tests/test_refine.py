"""Refining a search: look again at what the first pass ran out of calls to price.

The first pass hides itineraries whose legs it could not price, and measurement showed most
of those were starved of per-source calls rather than genuinely unpriceable (Uppsala->Göteborg,
2026-08-10: 17 hidden at the base allowance, 5 at double, and nothing beyond). What is pinned
here is the mechanism that lets a caller act on that: the count is a number rather than prose,
the larger allowance is per call rather than per orchestrator, and a refine leaves every other
search in the process untouched.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from tripps.config import CacheTTL, PricingBudget
from tripps.interfaces import HealthState, PriceAdapter, SourceHealth
from tripps.models import (
    ADULT,
    Itinerary,
    Leg,
    Passenger,
    PriceConfidence,
    Quote,
    SearchConstraints,
    Stop,
    TransportMode,
)
from tripps.pricing.base import BudgetExceeded
from tripps.pricing.orchestrator import PricingOrchestrator

TZ = ZoneInfo("Europe/Stockholm")
DAY = date(2026, 8, 10)
A = Stop(id="A", name="A", lat=59.33, lon=18.06)
B = Stop(id="B", name="B", lat=57.71, lon=11.97)


class BudgetedAdapter(PriceAdapter):
    """Prices a leg only while its allowance lasts, like a real source under a CallBudget."""

    name = "stub"
    modes = frozenset({TransportMode.TRAIN})

    def __init__(self, price_ore: int = 10_000) -> None:
        self.price_ore = price_ore
        self.calls = 0
        self._budget = None

    def _price_for(self, leg: Leg) -> int:
        # Distinct per departure, so `collapse_equivalent` cannot fold these itineraries into
        # one and quietly change the counts under test.
        return self.price_ore + leg.departure.hour * 100

    def set_budget(self, budget) -> None:
        self._budget = budget

    def supports(self, leg: Leg) -> bool:
        return leg.mode is TransportMode.TRAIN

    async def quote_leg(self, leg: Leg, passenger: Passenger = ADULT) -> Quote:
        if self._budget is not None:
            try:
                self._budget.consume(self.name)
            except BudgetExceeded as exc:
                return Quote.unavailable(source=self.name, note=str(exc))
        self.calls += 1
        return Quote(
            source=self.name, amount_ore=self._price_for(leg), confidence=PriceConfidence.EXACT
        )

    async def health(self) -> SourceHealth:
        return SourceHealth(self.name, HealthState.OK)


def _itinerary(hour: int) -> Itinerary:
    """One train an hour, each to its own destination.

    Distinct destinations on purpose: `collapse_equivalent` folds same-shape journeys that
    cost the same, and unpriced ones all cost None - so identical shapes would collapse into
    a single hidden option and the counts here would measure that instead of the budget.
    """
    return Itinerary(
        legs=[
            Leg(
                from_stop=A,
                to_stop=Stop(id=f"B{hour}", name=f"B{hour}", lat=57.71, lon=11.97),
                mode=TransportMode.TRAIN,
                operator="SJ",
                departure=datetime(2026, 8, 10, hour, 0, tzinfo=TZ),
                arrival=datetime(2026, 8, 10, hour + 4, 0, tzinfo=TZ),
                service_ref=f"T{hour}",
            )
        ]
    )


def _budget(calls: int, **kw) -> PricingBudget:
    return PricingBudget(
        max_calls_per_source_per_search=calls, min_interval_seconds=0.0, **kw
    )


# --- the refined budget ----------------------------------------------------


def test_refined_multiplies_the_call_allowance_and_extends_the_deadline():
    base = _budget(12, refine_call_multiplier=2, refine_timeout_seconds=90.0)
    refined = base.refined()
    assert refined.max_calls_per_source_per_search == 24
    assert refined.phase2_timeout_seconds == 90.0
    # A refine prices strictly more legs, so its deadline must not be the first pass's.
    assert refined.phase2_timeout_seconds > base.phase2_timeout_seconds


def test_refining_does_not_mutate_the_budget_other_searches_share():
    base = _budget(12)
    base.refined()
    assert base.max_calls_per_source_per_search == 12


def test_the_multiplier_is_configurable():
    assert _budget(10, refine_call_multiplier=4).refined().max_calls_per_source_per_search == 40


# --- the count, as a number -------------------------------------------------


async def test_hidden_options_is_counted_not_just_narrated():
    """The UI offers a refine off this number; parsing the English warning would be brittle."""
    orchestrator = PricingOrchestrator(
        [BudgetedAdapter()], None, budget=_budget(2), ttl=CacheTTL()
    )
    result = await orchestrator.price(
        [_itinerary(h) for h in (6, 8, 10, 12, 14)],
        SearchConstraints(),
        max_results=10,
    )
    assert result.hidden_options == 3
    assert any("3 option(s) with at least one unpriced leg were hidden" in w
               for w in result.warnings)
    assert len(result.itineraries) == 2


async def test_nothing_hidden_when_the_allowance_covers_everything():
    orchestrator = PricingOrchestrator(
        [BudgetedAdapter()], None, budget=_budget(12), ttl=CacheTTL()
    )
    result = await orchestrator.price(
        [_itinerary(h) for h in (6, 8, 10)], SearchConstraints(), max_results=10
    )
    assert result.hidden_options == 0
    assert len(result.itineraries) == 3


# --- the per-call budget override -------------------------------------------


async def test_a_larger_budget_passed_per_call_recovers_the_hidden_options():
    adapter = BudgetedAdapter()
    orchestrator = PricingOrchestrator([adapter], None, budget=_budget(2), ttl=CacheTTL())
    candidates = [_itinerary(h) for h in (6, 8, 10, 12, 14)]

    first = await orchestrator.price(candidates, SearchConstraints(), max_results=10)
    assert first.hidden_options == 3

    refined = await orchestrator.price(
        candidates,
        SearchConstraints(),
        max_results=10,
        budget=orchestrator.budget.refined().model_copy(
            update={"max_calls_per_source_per_search": 12}
        ),
    )
    assert refined.hidden_options == 0
    assert len(refined.itineraries) == 5


async def test_the_override_does_not_leak_into_the_next_search():
    """The orchestrator is shared by every search in the process; a refine is one call's worth."""
    orchestrator = PricingOrchestrator(
        [BudgetedAdapter()], None, budget=_budget(2), ttl=CacheTTL()
    )
    candidates = [_itinerary(h) for h in (6, 8, 10, 12, 14)]

    await orchestrator.price(
        candidates, SearchConstraints(), max_results=10, budget=_budget(12)
    )
    after = await orchestrator.price(candidates, SearchConstraints(), max_results=10)
    assert after.hidden_options == 3
    assert orchestrator.budget.max_calls_per_source_per_search == 2


@pytest.mark.parametrize("candidates_cap", [2, 4])
async def test_the_candidate_cap_is_taken_from_the_call_budget_too(candidates_cap):
    orchestrator = PricingOrchestrator(
        [BudgetedAdapter()], None, budget=_budget(12), ttl=CacheTTL()
    )
    result = await orchestrator.price(
        [_itinerary(h) for h in (6, 8, 10, 12, 14)],
        SearchConstraints(),
        max_results=10,
        budget=_budget(12).model_copy(update={"max_candidates_to_price": candidates_cap}),
    )
    assert any(f"(cap: {candidates_cap})" in w for w in result.warnings)
