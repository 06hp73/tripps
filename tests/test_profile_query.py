"""Regression: a cheaper-but-later departure must survive routing and reach pricing.

The bug this locks shut, observed against the real feed on 2026-07-13: FlixBus Stockholm ->
Goteborg costs 800 SEK at 07:30 and 420 SEK at 22:55. The routing price *floor* is derived
from distance and operator, so it is identical for both departures. With only (arrival,
price) as criteria, the night bus was "later arrival, same floor" and was pruned before
phase 2 could discover it was half the price. The planner claimed 800 SEK was the cheapest
way to cross Sweden. It was not.

Departure time is therefore a third, maximized Pareto criterion.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tripps.config import PricingBudget
from tripps.db import Database
from tripps.interfaces import PriceAdapter
from tripps.models import ADULT, Leg, Passenger, PriceConfidence, Quote, Stop, TransportMode
from tripps.pricing.orchestrator import PricingOrchestrator
from tripps.routing.floors import PriceFloorModel
from tripps.routing.journey import (
    collapse_equivalent,
    label_to_itinerary,
    pattern_key,
    spread_by_departure,
)
from tripps.routing.mcraptor import RaptorQuery, run_mcraptor
from tripps.routing.timetable import INFINITY, RouteInfo, TimetableBuilder, Trip
from tripps.search import Planner, SearchOptions

TZ = ZoneInfo("Europe/Stockholm")


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "p.sqlite3")
    yield database
    database.close()


def _hhmm(h, m=0) -> int:
    return h * 3600 + m * 60


STO = Stop(id="STO", name="Stockholm", lat=59.33, lon=18.06)
GBG = Stop(id="GBG", name="Goteborg", lat=57.71, lon=11.97)

#: Real FlixBus fares for 2026-07-13, from the live endpoint.
REAL_FARES = {"07:30": 800_00, "10:50": 640_00, "15:05": 570_00, "22:55": 420_00}


def _flix_timetable():
    b = TimetableBuilder()
    b.add_stop(STO)
    b.add_stop(GBG)
    info = RouteInfo(id="FLIX", mode=TransportMode.BUS, operator="FlixBus")
    for dep_h, dep_m, arr in ((7, 30, _hhmm(13, 45)), (10, 50, _hhmm(17, 25)),
                              (15, 5, _hhmm(21, 35)), (22, 55, _hhmm(29, 45))):
        dep = _hhmm(dep_h, dep_m)
        b.add_trip(info, ["STO", "GBG"], Trip(id=f"flix-{dep_h:02d}{dep_m:02d}",
                                              arrivals=[dep, arr], departures=[dep, arr]))
    return b.build()


class FaresByDeparture(PriceAdapter):
    """Prices a bus by the clock, the way a yield-managed operator really does."""

    name = "flix-stub"
    modes = frozenset({TransportMode.BUS})

    def __init__(self, fares: dict[str, int]) -> None:
        self.fares = fares
        self.calls = 0

    def supports(self, leg: Leg) -> bool:
        return leg.mode is TransportMode.BUS

    async def quote_leg(self, leg: Leg, passenger: Passenger = ADULT) -> Quote:
        self.calls += 1
        key = leg.departure.strftime("%H:%M")
        amount = self.fares.get(key)
        if amount is None:
            return Quote.unavailable(self.name)
        return Quote(source=self.name, amount_ore=amount, confidence=PriceConfidence.EXACT)


# --- the router keeps later, cheaper departures -----------------------------


def test_range_query_returns_every_departure():
    tt = _flix_timetable()
    res = run_mcraptor(
        tt, PriceFloorModel(),
        RaptorQuery(origins=[(tt.index_of("STO"), _hhmm(6))], targets={tt.index_of("GBG")}),
    )
    departures = sorted(lbl.departure for lbl in res.labels)
    assert departures == [_hhmm(7, 30), _hhmm(10, 50), _hhmm(15, 5), _hhmm(22, 55)]


def test_without_profile_only_the_earliest_survives():
    """Demonstrates the bug this module exists to prevent."""
    tt = _flix_timetable()
    res = run_mcraptor(
        tt, PriceFloorModel(),
        RaptorQuery(
            origins=[(tt.index_of("STO"), _hhmm(6))],
            targets={tt.index_of("GBG")},
            profile=False,
        ),
    )
    assert len(res.labels) == 1
    assert res.labels[0].departure == _hhmm(7, 30)  # the 800 SEK bus, and nothing else


def test_origin_label_has_no_departure_until_it_boards():
    tt = _flix_timetable()
    res = run_mcraptor(
        tt, PriceFloorModel(),
        RaptorQuery(origins=[(tt.index_of("STO"), _hhmm(6))], targets={tt.index_of("GBG")}),
    )
    assert all(lbl.departure != INFINITY for lbl in res.labels)


def test_profile_window_bounds_the_enumeration():
    tt = _flix_timetable()
    res = run_mcraptor(
        tt, PriceFloorModel(),
        RaptorQuery(
            origins=[(tt.index_of("STO"), _hhmm(6))],
            targets={tt.index_of("GBG")},
            profile_window_seconds=6 * 3600,  # only until noon
        ),
    )
    assert sorted(lbl.departure for lbl in res.labels) == [_hhmm(7, 30), _hhmm(10, 50)]


def test_max_departures_per_route_caps_the_frontier():
    tt = _flix_timetable()
    res = run_mcraptor(
        tt, PriceFloorModel(),
        RaptorQuery(
            origins=[(tt.index_of("STO"), _hhmm(6))],
            targets={tt.index_of("GBG")},
            max_departures_per_route=2,
            # A coach route reads the intercity cap; pin both so the test exercises the
            # capping mechanism itself rather than the (deliberately high) default.
            max_departures_per_route_intercity=2,
        ),
    )
    assert len(res.labels) == 2


def test_mid_journey_waiting_for_a_later_bus_is_still_pruned():
    """A later trip on a connecting leg only arrives later at the same price, so it must
    not multiply the frontier."""
    b = TimetableBuilder()
    b.add_stop(STO)
    b.add_stop(Stop(id="NRK", name="Norrkoping", lat=58.596, lon=16.183))
    b.add_stop(GBG)
    leg1 = RouteInfo(id="L1", mode=TransportMode.TRAIN, operator="SJ")
    b.add_trip(leg1, ["STO", "NRK"], Trip(id="t1", arrivals=[_hhmm(8), _hhmm(9)],
                                          departures=[_hhmm(8), _hhmm(9)]))
    leg2 = RouteInfo(id="L2", mode=TransportMode.TRAIN, operator="SJ")
    for hour in (10, 12, 14, 16):
        b.add_trip(leg2, ["NRK", "GBG"], Trip(id=f"t2-{hour}",
                                              arrivals=[_hhmm(hour), _hhmm(hour + 3)],
                                              departures=[_hhmm(hour), _hhmm(hour + 3)]))
    tt = b.build()
    res = run_mcraptor(
        tt, PriceFloorModel(),
        RaptorQuery(origins=[(tt.index_of("STO"), _hhmm(7))], targets={tt.index_of("GBG")}),
    )
    # One journey: board the 08:00, connect to the first onward train. Waiting for later
    # connections never helps.
    assert len(res.labels) == 1
    assert res.labels[0].arrival == _hhmm(13)


# --- candidate selection ----------------------------------------------------


def _itineraries():
    tt = _flix_timetable()
    res = run_mcraptor(
        tt, PriceFloorModel(),
        RaptorQuery(origins=[(tt.index_of("STO"), _hhmm(6))], targets={tt.index_of("GBG")}),
    )
    return [label_to_itinerary(tt, lbl, date(2026, 7, 13)) for lbl in res.labels]


def test_same_shape_different_departures_share_a_pattern_key():
    itins = _itineraries()
    assert len({pattern_key(i) for i in itins}) == 1


def test_spread_samples_across_the_day_not_just_the_morning():
    itins = _itineraries()
    kept = spread_by_departure(itins, max_per_pattern=2)
    hours = sorted(i.departure.hour for i in kept)
    assert hours == [7, 22], "the first and last departures, not the first two"


def test_spread_keeps_everything_when_under_the_cap():
    itins = _itineraries()
    assert len(spread_by_departure(itins, max_per_pattern=10)) == len(itins)


def test_collapse_folds_identical_priced_journeys_but_keeps_distinct_prices():
    itins = _itineraries()
    cheap = itins[0].model_copy(
        update={
            "legs": [
                itins[0].legs[0].model_copy(
                    update={"quote": Quote(source="s", amount_ore=100, confidence=PriceConfidence.EXACT)}
                )
            ]
        }
    )
    dear = itins[1].model_copy(
        update={
            "legs": [
                itins[1].legs[0].model_copy(
                    update={"quote": Quote(source="s", amount_ore=900, confidence=PriceConfidence.EXACT)}
                )
            ]
        }
    )
    same_as_cheap = itins[2].model_copy(
        update={
            "legs": [
                itins[2].legs[0].model_copy(
                    update={"quote": Quote(source="s", amount_ore=100, confidence=PriceConfidence.EXACT)}
                )
            ]
        }
    )
    collapsed = collapse_equivalent([cheap, dear, same_as_cheap])
    assert len(collapsed) == 2
    assert sorted(i.total_price_ore for i in collapsed) == [100, 900]


# --- end to end: the night bus wins -----------------------------------------


async def test_cheapest_departure_is_found_and_ranked_first(db):
    """The whole point. Four FlixBus departures, real fares; the 22:55 must win."""
    tt = _flix_timetable()
    adapter = FaresByDeparture(REAL_FARES)
    orchestrator = PricingOrchestrator(
        [adapter], db, budget=PricingBudget(min_interval_seconds=0.0), floors=PriceFloorModel()
    )
    planner = Planner(tt, orchestrator, db=db, options=SearchOptions(include_flights=False))

    response, stats = await planner.search(
        "Stockholm", "Goteborg", date(2026, 7, 13),
        departure_after=datetime(2026, 7, 13, 6, 0, tzinfo=TZ),
    )

    assert stats.candidates == 4, "every departure must reach the pricing stage"
    best = response.itineraries[0]
    assert best.departure.strftime("%H:%M") == "22:55"
    assert best.total_price_ore == 420_00

    # And the expensive morning bus is still offered, just not first.
    prices = [i.total_price_ore for i in response.itineraries]
    assert prices == sorted(prices)
    assert 800_00 in prices


async def test_cheapest_is_found_even_when_it_arrives_last(db):
    """The night bus arrives at 05:45 the next morning: latest arrival, lowest price."""
    tt = _flix_timetable()
    orchestrator = PricingOrchestrator(
        [FaresByDeparture(REAL_FARES)], db, budget=PricingBudget(min_interval_seconds=0.0)
    )
    planner = Planner(tt, orchestrator, db=db, options=SearchOptions(include_flights=False))
    response, _ = await planner.search(
        "Stockholm", "Goteborg", date(2026, 7, 13),
        departure_after=datetime(2026, 7, 13, 6, 0, tzinfo=TZ),
    )
    best = response.itineraries[0]
    assert best.arrival == max(i.arrival for i in response.itineraries)
    assert best.total_price_ore == min(i.total_price_ore for i in response.itineraries)


async def test_duration_constraint_overrides_price(db):
    """"Cheapest subject to constraints": a tight cap must beat a cheap fare.

    Durations: 07:30 is 6h15m, 15:05 is 6h30m, 10:50 is 6h35m, 22:55 is 6h50m. A 6h20m cap
    leaves only the 07:30 at 800 SEK - the priciest option, and the correct answer once the
    user says they will not sit on a bus for longer.
    """
    tt = _flix_timetable()
    orchestrator = PricingOrchestrator(
        [FaresByDeparture(REAL_FARES)], db, budget=PricingBudget(min_interval_seconds=0.0)
    )
    planner = Planner(tt, orchestrator, db=db, options=SearchOptions(include_flights=False))

    from tripps.models import SearchConstraints

    response, _ = await planner.search(
        "Stockholm", "Goteborg", date(2026, 7, 13),
        constraints=SearchConstraints(max_duration_seconds=6 * 3600 + 20 * 60),
        departure_after=datetime(2026, 7, 13, 6, 0, tzinfo=TZ),
    )
    departures = {i.departure.strftime("%H:%M") for i in response.itineraries}
    assert departures == {"07:30"}
    assert response.itineraries[0].total_price_ore == 800_00

    # Without the cap, the cheap night bus returns.
    unbounded, _ = await planner.search(
        "Stockholm", "Goteborg", date(2026, 7, 13),
        departure_after=datetime(2026, 7, 13, 6, 0, tzinfo=TZ),
    )
    assert unbounded.itineraries[0].departure.strftime("%H:%M") == "22:55"


# --- itinerary reconstruction ----------------------------------------------


def test_leading_walk_is_delayed_to_the_moment_of_boarding():
    """A journey that walks to a car park and waits six hours must not claim to depart at
    midnight. RAPTOR relaxes footpaths as soon as a stop is reachable; the traveller does
    not leave home then."""
    from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip

    b = TimetableBuilder()
    b.add_stop(Stop(id="HOME", name="Stockholm C", lat=59.3300, lon=18.0590))
    b.add_stop(Stop(id="LOT", name="Car park", lat=59.3340, lon=18.0590))
    b.add_stop(Stop(id="ARN", name="Arlanda", lat=59.6485, lon=17.9288))
    info = RouteInfo(id="CAR", mode=TransportMode.FREERIDER, operator="hertz-freerider")
    b.add_trip(
        info, ["LOT", "ARN"],
        Trip(id="car", arrivals=[_hhmm(6), _hhmm(6, 31)],
             departures=[_hhmm(6), _hhmm(6, 31)], precomputed_fare_ore=0),
    )
    b.add_transfer("HOME", "LOT", 600)
    tt = b.build()

    res = run_mcraptor(
        tt, PriceFloorModel(),
        RaptorQuery(origins=[(tt.index_of("HOME"), 0)], targets={tt.index_of("ARN")}),
    )
    assert res.labels
    itin = label_to_itinerary(tt, res.labels[0], date(2026, 8, 3))

    walk = itin.legs[0]
    assert walk.mode is TransportMode.WALK
    assert walk.arrival == itin.legs[1].departure, "the walk ends exactly at boarding"
    assert walk.arrival - walk.departure == timedelta(seconds=600), "its duration is unchanged"
    assert itin.departure.strftime("%H:%M") == "05:50"
    assert itin.duration_seconds == 41 * 60  # 10 min walk + 31 min drive, no phantom wait


def test_walk_between_vehicles_keeps_its_real_waiting_time():
    """Only the walk before the first vehicle moves. A connection's slack is real waiting."""
    from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip

    b = TimetableBuilder()
    b.add_stop(Stop(id="A", name="A", lat=59.33, lon=18.06))
    b.add_stop(Stop(id="B", name="B", lat=58.60, lon=16.18))
    b.add_stop(Stop(id="C", name="C", lat=58.60, lon=16.19))
    b.add_stop(Stop(id="D", name="D", lat=57.71, lon=11.97))
    r1 = RouteInfo(id="R1", mode=TransportMode.TRAIN, operator="SJ")
    b.add_trip(r1, ["A", "B"], Trip(id="t1", arrivals=[_hhmm(8), _hhmm(9)], departures=[_hhmm(8), _hhmm(9)]))
    r2 = RouteInfo(id="R2", mode=TransportMode.BUS, operator="FlixBus")
    b.add_trip(r2, ["C", "D"], Trip(id="t2", arrivals=[_hhmm(12), _hhmm(15)], departures=[_hhmm(12), _hhmm(15)]))
    b.add_transfer("B", "C", 300)
    tt = b.build()

    res = run_mcraptor(
        tt, PriceFloorModel(),
        RaptorQuery(origins=[(tt.index_of("A"), _hhmm(7))], targets={tt.index_of("D")}),
    )
    itin = label_to_itinerary(tt, res.labels[0], date(2026, 7, 13))
    walk = next(leg for leg in itin.legs if leg.mode is TransportMode.WALK)
    assert walk.departure.strftime("%H:%M") == "09:00", "walk starts when the train arrives"
    assert itin.departure.strftime("%H:%M") == "08:00"
