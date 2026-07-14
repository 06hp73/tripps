"""Split-ticketing: sub-leg construction, and the advisory hint the orchestrator attaches.

The invariant under test is the ethos one: a split is surfaced as a saving *tip* only, never as
the itinerary's authoritative price - the total tripps stands behind stays the bookable through
fare even when a cheaper two-ticket split exists.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tripps.config import PricingBudget
from tripps.interfaces import HealthState, SourceHealth
from tripps.models import Itinerary, Leg, PriceConfidence, Quote, Stop, TransportMode
from tripps.pricing.orchestrator import PricingOrchestrator
from tripps.pricing.split import SPLIT_STATIONS, sub_legs

STHLM = Stop(id="740000001", name="Stockholm C", lat=59.330, lon=18.059)
MALMO = Stop(id="740000002", name="Malmö C", lat=55.609, lon=13.000)
ALVESTA_ID = "740000004"  # a curated hub


def _dt(h: int, m: int = 0) -> datetime:
    return datetime(2026, 8, 3, h, m)


def _through_leg() -> Leg:
    """A Stockholm->Malmö SJ train that calls at Alvesta, with per-stop timing."""
    return Leg(
        from_stop=STHLM,
        to_stop=MALMO,
        mode=TransportMode.TRAIN,
        operator="SJ",
        departure=_dt(8, 0),
        arrival=_dt(12, 0),
        service_ref="sj-577",
        via_stop_ids=(STHLM.id, ALVESTA_ID, MALMO.id),
        via_departures=(_dt(8, 0), _dt(10, 5), _dt(12, 0)),
        via_arrivals=(_dt(8, 0), _dt(10, 0), _dt(12, 0)),
    )


class FakeSJ:
    """An SJ adapter whose fare depends only on (origin, destination), so a test can make the
    split cheaper (or not) than the through fare."""

    name = "sj"
    provides_price = True

    def __init__(self, fares: dict[tuple[str, str], int]):
        self.fares = fares

    def supports(self, leg: Leg) -> bool:
        return leg.mode is TransportMode.TRAIN

    async def quote_leg(self, leg: Leg) -> Quote:
        amount = self.fares.get((leg.from_stop.id, leg.to_stop.id))
        if amount is None:
            return Quote.unavailable(source="sj", note="no fare")
        return Quote(source="sj", amount_ore=amount, confidence=PriceConfidence.EXACT)

    async def health(self) -> SourceHealth:
        return SourceHealth(self.name, HealthState.OK)

    async def aclose(self) -> None:
        pass


def _orch(fares: dict[tuple[str, str], int]) -> PricingOrchestrator:
    return PricingOrchestrator(
        [FakeSJ(fares)], None, budget=PricingBudget(min_interval_seconds=0.0)
    )


# --- sub-leg construction ------------------------------------------------------------------


def test_sub_legs_break_at_hub_with_same_train_timing():
    parts = sub_legs(_through_leg())
    assert len(parts) == 1
    name, first, second = parts[0]
    assert name == SPLIT_STATIONS[ALVESTA_ID].name
    # First half: origin -> hub, boards when the train boards, alights when it reaches the hub.
    assert first.from_stop.id == STHLM.id
    assert first.to_stop.id == ALVESTA_ID
    assert first.departure == _dt(8, 0)
    assert first.arrival == _dt(10, 0)
    # Second half: hub -> destination, departs the hub at the train's own hub departure.
    assert second.from_stop.id == ALVESTA_ID
    assert second.to_stop.id == MALMO.id
    assert second.departure == _dt(10, 5)
    assert second.arrival == _dt(12, 0)
    assert first.quote is None and second.quote is None


def test_no_sub_legs_without_via_timing():
    leg = _through_leg().model_copy(update={"via_departures": (), "via_arrivals": ()})
    assert sub_legs(leg) == []


def test_no_sub_legs_when_no_interior_hub():
    # Endpoints only, no hub in between.
    leg = _through_leg().model_copy(
        update={
            "via_stop_ids": (STHLM.id, MALMO.id),
            "via_departures": (_dt(8, 0), _dt(12, 0)),
            "via_arrivals": (_dt(8, 0), _dt(12, 0)),
        }
    )
    assert sub_legs(leg) == []


# --- the orchestrator advisory -------------------------------------------------------------


@pytest.mark.asyncio
async def test_hint_attached_when_split_is_cheaper_and_total_unchanged():
    orch = _orch(
        {
            (STHLM.id, MALMO.id): 40000,  # 400 kr through
            (STHLM.id, ALVESTA_ID): 19500,  # 195 kr
            (ALVESTA_ID, MALMO.id): 17500,  # 175 kr  -> 370 kr split, saves 30 kr
        }
    )
    result = await orch.price([Itinerary(legs=[_through_leg()])])

    assert len(result.itineraries) == 1
    itin = result.itineraries[0]
    leg = itin.legs[0]
    assert leg.quote.split_hint is not None
    assert "Alvesta" in leg.quote.split_hint
    assert "30 kr" in leg.quote.split_hint  # the saving
    # The authoritative numbers never move: the leg amount and the total stay the through fare.
    assert leg.quote.amount_ore == 40000
    assert itin.total_price_ore == 40000
    assert result.floor_violations == 0


@pytest.mark.asyncio
async def test_no_hint_when_split_is_not_cheaper():
    orch = _orch(
        {
            (STHLM.id, MALMO.id): 40000,
            (STHLM.id, ALVESTA_ID): 25000,
            (ALVESTA_ID, MALMO.id): 25000,  # 500 kr split > 400 kr through
        }
    )
    result = await orch.price([Itinerary(legs=[_through_leg()])])
    assert result.itineraries[0].legs[0].quote.split_hint is None


@pytest.mark.asyncio
async def test_no_hint_for_trivial_saving_below_threshold():
    orch = _orch(
        {
            (STHLM.id, MALMO.id): 40000,
            (STHLM.id, ALVESTA_ID): 20000,
            (ALVESTA_ID, MALMO.id): 19600,  # saves 4 kr, under the 5 kr floor
        }
    )
    result = await orch.price([Itinerary(legs=[_through_leg()])])
    assert result.itineraries[0].legs[0].quote.split_hint is None


@pytest.mark.asyncio
async def test_disabled_by_config():
    orch = PricingOrchestrator(
        [FakeSJ({(STHLM.id, MALMO.id): 40000, (STHLM.id, ALVESTA_ID): 100, (ALVESTA_ID, MALMO.id): 100})],
        None,
        budget=PricingBudget(min_interval_seconds=0.0, enable_split_tickets=False),
    )
    result = await orch.price([Itinerary(legs=[_through_leg()])])
    assert result.itineraries[0].legs[0].quote.split_hint is None
