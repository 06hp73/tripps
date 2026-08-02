"""Charging only the stretch a held period card does not reach.

A card does not stop being valid because the train carries on past its border. Riding
Halmstad->Göteborg on a Hallandstrafiken card, the holder travels free to the edge of Halland
and buys a ticket for the remainder: 90 SEK (Åsa->Göteborg), not the 195 the whole leg costs.

Two things are load-bearing and pinned here. The reduction must reach *ranking*, because it
changes which journey is cheapest - at 195 the coach wins that corridor, at 90 the train does.
And it must never reach the *cache*: the quote cache is keyed on the leg and the traveller,
not on which cards happen to be registered, so a holder's remainder fare stored there would be
served to everyone.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tripps.config import CacheTTL, PricingBudget
from tripps.db import Database
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
from tripps.passes import PassAdapter, PassCoverage, TravelCard
from tripps.pricing.orchestrator import CARD_REMAINDER_FARE_CLASS, PricingOrchestrator

TZ = ZoneInfo("Europe/Stockholm")

# Halmstad -> Varberg -> Åsa -> Kungsbacka -> Göteborg, the real corridor in miniature.
PATH = [
    ("740000080", "Halmstad C"),
    ("740000110", "Varberg"),
    ("740001604", "Åsa"),
    ("740000161", "Kungsbacka"),
    ("740000002", "Göteborg C"),
]
STOPS = {sid: Stop(id=sid, name=name, lat=57.0, lon=12.0) for sid, name in PATH}

#: Halland reaches Åsa; Kungsbacka and Göteborg are Västtrafik's, matching what the national
#: feed's own agency-to-stop map says.
HALLAND = TravelCard(
    id="hallandstrafiken",
    name="Hallandstrafiken",
    region="Halland",
    honored_operators=frozenset({"Öresundståg"}),
    coverage_model="region-stops",
    region_agencies=frozenset({"Hallandstrafiken"}),
)
AGENCY_STOPS = {"Hallandstrafiken": ["740000080", "740000110", "740001604"]}


def _leg(from_index: int = 0, to_index: int = 4) -> Leg:
    ids = [sid for sid, _ in PATH][from_index : to_index + 1]
    times = [datetime(2026, 8, 10, 6 + i, 0, tzinfo=TZ) for i in range(len(ids))]
    return Leg(
        from_stop=STOPS[ids[0]],
        to_stop=STOPS[ids[-1]],
        mode=TransportMode.TRAIN,
        operator="Öresundståg",
        departure=times[0],
        arrival=times[-1],
        service_ref="OT-1",
        via_stop_ids=tuple(ids),
        via_departures=tuple(times),
        via_arrivals=tuple(times),
    )


def _coverage() -> PassCoverage:
    return PassCoverage(None, cards={HALLAND.id: HALLAND}, agency_stops=AGENCY_STOPS)


def _pass_adapter(held=("hallandstrafiken",)) -> PassAdapter:
    adapter = PassAdapter(agency_stops=AGENCY_STOPS)
    adapter._coverage = _coverage()
    adapter._held = list(held)
    return adapter


class FaresAdapter(PriceAdapter):
    """Prices any train leg from a table of (from_id, to_id) -> öre."""

    name = "tora"
    modes = frozenset({TransportMode.TRAIN})

    def __init__(self, fares: dict[tuple[str, str], int]) -> None:
        self.fares = fares
        self.asked: list[tuple[str, str]] = []

    def supports(self, leg: Leg) -> bool:
        return leg.mode is TransportMode.TRAIN

    async def quote_leg(self, leg: Leg, passenger: Passenger = ADULT) -> Quote:
        key = (leg.from_stop.id, leg.to_stop.id)
        self.asked.append(key)
        amount = self.fares.get(key)
        if amount is None:
            return Quote.unavailable(source=self.name, note="no fare for this pair")
        return Quote(source=self.name, amount_ore=amount, confidence=PriceConfidence.EXACT)

    async def health(self) -> SourceHealth:
        return SourceHealth(self.name, HealthState.OK)


FARES = {
    ("740000080", "740000002"): 19_500,  # Halmstad -> Göteborg, the whole ride
    ("740001604", "740000002"): 9_000,   # Åsa -> Göteborg, the uncovered remainder
}


# --- how far the cards reach ------------------------------------------------


def test_covered_run_stops_at_the_card_border():
    boundary, cards = _coverage().covered_run(["hallandstrafiken"], _leg())
    assert [sid for sid, _ in PATH][boundary] == "740001604"  # Åsa, the last Halland stop
    assert [c.id for c in cards] == ["hallandstrafiken"]


def test_no_run_when_the_card_covers_the_whole_leg():
    """That is `covers`'s business - this is only about the part that is left to pay."""
    assert _coverage().covered_run(["hallandstrafiken"], _leg(0, 2)) is None


def test_no_run_when_the_journey_starts_outside_the_card():
    """Kungsbacka -> Göteborg is entirely Västtrafik's; a Halland card reaches none of it."""
    assert _coverage().covered_run(["hallandstrafiken"], _leg(3, 4)) is None


def test_no_run_without_a_stop_path():
    """Fails closed: with no path there is no way to know where the train calls."""
    bare = _leg().model_copy(update={"via_stop_ids": (), "via_departures": (), "via_arrivals": ()})
    assert _coverage().covered_run(["hallandstrafiken"], bare) is None


def test_no_run_for_an_operator_the_card_does_not_honour():
    other = _leg().model_copy(update={"operator": "SJ"})
    assert _coverage().covered_run(["hallandstrafiken"], other) is None


def test_no_run_without_a_held_card():
    assert _coverage().covered_run([], _leg()) is None


# --- what the traveller is charged ------------------------------------------


def _orchestrator(db=None, held=("hallandstrafiken",), fares=None):
    paid = FaresAdapter(dict(FARES if fares is None else fares))
    orchestrator = PricingOrchestrator(
        [_pass_adapter(held), paid],
        db,
        budget=PricingBudget(min_interval_seconds=0.0),
        ttl=CacheTTL(),
    )
    return orchestrator, paid


async def test_the_holder_is_charged_only_the_remainder():
    orchestrator, paid = _orchestrator()
    # The planner supplies this; without it a note can only name the boundary by its stop id.
    result = await orchestrator.price(
        [Itinerary(legs=[_leg()])], SearchConstraints(), stop_resolver=STOPS.get
    )

    quote = result.itineraries[0].legs[0].quote
    assert quote.amount_ore == 9_000
    assert quote.fare_class == CARD_REMAINDER_FARE_CLASS
    assert quote.note == (
        "Halmstad C → Åsa is covered by your Hallandstrafiken period ticket; "
        "only Åsa → Göteborg C is charged."
    )
    assert ("740001604", "740000002") in paid.asked


async def test_the_reduction_changes_which_journey_is_cheapest():
    """The point of doing this during pricing: at 195 the coach wins, at 90 the train does."""
    orchestrator, _ = _orchestrator()
    coach = Itinerary(
        legs=[
            _leg().model_copy(
                update={
                    "mode": TransportMode.BUS,
                    "operator": "Flixbus",
                    "service_ref": "FB-1",
                    "quote": Quote(
                        source="flixbus", amount_ore=16_000, confidence=PriceConfidence.EXACT
                    ),
                }
            )
        ]
    )
    result = await orchestrator.price(
        [coach, Itinerary(legs=[_leg()])], SearchConstraints(), max_results=5
    )
    assert result.itineraries[0].legs[0].mode is TransportMode.TRAIN
    assert result.itineraries[0].total_price_ore == 9_000


async def test_without_the_card_the_full_fare_stands():
    orchestrator, _ = _orchestrator(held=())
    result = await orchestrator.price([Itinerary(legs=[_leg()])], SearchConstraints())
    assert result.itineraries[0].legs[0].quote.amount_ore == 19_500


async def test_the_through_fare_stands_when_the_remainder_is_no_cheaper():
    """Short-hop minimums can price the tail as dearly as the ride; then one ticket is better."""
    orchestrator, _ = _orchestrator(
        fares={**FARES, ("740001604", "740000002"): 19_500}
    )
    result = await orchestrator.price([Itinerary(legs=[_leg()])], SearchConstraints())
    quote = result.itineraries[0].legs[0].quote
    assert quote.amount_ore == 19_500
    assert quote.fare_class != CARD_REMAINDER_FARE_CLASS


async def test_the_full_fare_stands_when_the_remainder_cannot_be_priced():
    orchestrator, _ = _orchestrator(fares={("740000080", "740000002"): 19_500})
    result = await orchestrator.price([Itinerary(legs=[_leg()])], SearchConstraints())
    assert result.itineraries[0].legs[0].quote.amount_ore == 19_500


# --- and never into the cache -----------------------------------------------


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "cards.sqlite3")
    yield database
    database.close()


async def test_the_cache_keeps_the_full_fare_not_the_holders(db):
    """The cache is not keyed on which cards are registered, so it must never learn one."""
    orchestrator, _ = _orchestrator(db)
    result = await orchestrator.price([Itinerary(legs=[_leg()])], SearchConstraints())
    assert result.itineraries[0].legs[0].quote.amount_ore == 9_000

    cached = db.get_quote("tora", _leg())
    assert cached is not None
    assert cached.amount_ore == 19_500


async def test_a_later_search_without_the_card_is_not_served_the_holders_price(db):
    holder, _ = _orchestrator(db)
    await holder.price([Itinerary(legs=[_leg()])], SearchConstraints())

    # Same database, no card registered: the cached full fare is what comes back.
    everyone_else, _ = _orchestrator(db, held=())
    result = await everyone_else.price([Itinerary(legs=[_leg()])], SearchConstraints())
    assert result.itineraries[0].legs[0].quote.amount_ore == 19_500
