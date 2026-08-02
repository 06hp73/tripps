"""The exhaustive SJ split scan: train discovery, point selection, and the chain optimiser.

The optimiser is the part that decides what a traveller is told to buy, so it is pinned
hardest: it must find the true cheapest chain over any number of breaks (not the best single
break), route around pairs SJ will not sell, and refuse to call a chain "cheaper" when a
single through ticket is as good - two tickets carry no protection across the break.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from tripps.models import Stop, TransportMode
from tripps.pricing.deepsplit import (
    MIN_SAVING_ORE,
    CallingPoint,
    ScanResult,
    Ticket,
    TrainRun,
    best_chain,
    choose_points,
    direct_runs,
)
from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip

TZ = ZoneInfo("Europe/Stockholm")
DAY = date(2026, 8, 10)


def _point(name: str, hour: int, *, board: bool = True, alight: bool = True) -> CallingPoint:
    moment = datetime(2026, 8, 10, hour, 0, tzinfo=TZ)
    return CallingPoint(
        stop=Stop(id=name, name=name, lat=59.0, lon=18.0),
        departure=moment,
        arrival=moment,
        can_board=board,
        can_alight=alight,
    )


def _run(*points: CallingPoint) -> TrainRun:
    return TrainRun(trip_id="T1", operator="SJ", headsign="Malmö", points=points)


# --- the optimiser ---------------------------------------------------------


def test_no_break_when_the_through_fare_is_the_cheapest_edge():
    prices = {(0, 1): 100, (1, 2): 100, (0, 2): 150}
    assert best_chain(prices, 3) == [(0, 2, 150)]


def test_finds_a_single_break_that_undercuts_the_through_fare():
    prices = {(0, 1): 60, (1, 2): 60, (0, 2): 150}
    assert best_chain(prices, 3) == [(0, 1, 60), (1, 2, 60)]


def test_beats_every_single_break_by_using_several():
    """The real shape: two breaks beat any one of them, which a hub-at-a-time probe misses."""
    prices = {
        (0, 3): 1000,           # through
        (0, 1): 300, (1, 3): 650,   # one break: 950
        (0, 2): 700, (2, 3): 280,   # the other: 980
        (1, 2): 350,                # both breaks: 300 + 350 + 280 = 930
    }
    chain = best_chain(prices, 4)
    assert [(a, b) for a, b, _ in chain] == [(0, 1), (1, 2), (2, 3)]
    assert sum(p for _, _, p in chain) == 930


def test_routes_around_pairs_sj_will_not_sell():
    """A missing pair is an absent edge, not a zero - this is what rescues an unsold through."""
    prices = {(0, 1): 400, (1, 2): 300}  # no (0, 2) at all
    assert best_chain(prices, 3) == [(0, 1, 400), (1, 2, 300)]


def test_no_chain_when_the_train_cannot_be_covered():
    assert best_chain({(0, 1): 100}, 3) == []
    assert best_chain({}, 4) == []
    assert best_chain({}, 1) == []


# --- when a chain is allowed to win ----------------------------------------


def _result(through: int | None, chain: list[int]) -> ScanResult:
    points = [_point(f"S{i}", 8 + i) for i in range(len(chain) + 1)]
    tickets = [Ticket(points[i], points[i + 1], price) for i, price in enumerate(chain)]
    return ScanResult(
        run=_run(*points),
        through_ore=through,
        tickets=tickets,
        scanned_points=len(points),
        total_points=len(points),
        pairs_priced=len(chain),
        pairs_unsellable=0,
        calls_made=0,
    )


def test_a_chain_must_beat_the_through_fare_by_a_real_margin():
    assert _result(100_000, [100_000 - MIN_SAVING_ORE]).chain_wins
    assert not _result(100_000, [100_000 - MIN_SAVING_ORE + 1]).chain_wins


def test_the_single_ticket_wins_a_tie():
    """Same money, one contract: nothing to gain and delay protection to lose."""
    tied = _result(50_000, [50_000])
    assert tied.saving_ore == 0
    assert not tied.chain_wins


def test_a_chain_wins_outright_when_no_through_fare_exists():
    rescued = _result(None, [30_000, 20_000])
    assert rescued.chain_ore == 50_000
    assert rescued.saving_ore is None
    assert rescued.chain_wins


def test_no_tickets_means_no_answer():
    assert not _result(None, []).chain_wins


# --- choosing what to scan -------------------------------------------------


def test_both_endpoints_are_always_scanned():
    run = _run(*[_point(f"S{i}", 8 + i) for i in range(6)])
    selection = choose_points(run, max_points=2)
    assert selection.indices == [0, 5]
    assert selection.capped == 4


def test_interior_points_are_thinned_evenly_when_capped():
    run = _run(*[_point(f"S{i}", 8 + i) for i in range(10)])
    chosen = choose_points(run, max_points=5).indices
    assert len(chosen) == 5
    assert chosen[0] == 0 and chosen[-1] == 9
    assert chosen == sorted(set(chosen))


def test_a_stop_nobody_may_use_is_never_a_break():
    """A set-down-only stop cannot be where a ticket starts, so it can bound nothing."""
    run = _run(
        _point("A", 8),
        _point("B", 9, board=False),
        _point("C", 10),
        _point("D", 11),
    )
    selection = choose_points(run, max_points=10)
    assert selection.indices == [0, 2, 3]
    # Blamed on the operator, not on the cap: raising --max-points would not add it back.
    assert selection.unusable == 1
    assert selection.capped == 0


def test_everything_is_scanned_when_the_cap_allows():
    run = _run(*[_point(f"S{i}", 8 + i) for i in range(4)])
    selection = choose_points(run, max_points=12)
    assert selection.indices == [0, 1, 2, 3]
    assert selection.capped == 0 and selection.unusable == 0


# --- finding the train -----------------------------------------------------


@pytest.fixture
def timetable():
    """Stockholm -> Nässjö -> Alvesta -> Malmö on one SJ train, with real GTFS stop ids."""
    builder = TimetableBuilder()
    stops = [
        Stop(id="740000001", name="Stockholm C", lat=59.330, lon=18.059),
        Stop(id="740000140", name="Nässjö C", lat=57.652, lon=14.694),
        Stop(id="740000004", name="Alvesta", lat=56.899, lon=14.556),
        Stop(id="740000003", name="Malmö C", lat=55.609, lon=13.000),
    ]
    for stop in stops:
        builder.add_stop(stop)
    builder.add_trip(
        RouteInfo(id="sj-south", mode=TransportMode.TRAIN, operator="SJ"),
        [s.id for s in stops],
        Trip(
            id="T1",
            arrivals=[7 * 3600, 10 * 3600, 11 * 3600, 12 * 3600],
            departures=[7 * 3600, 10 * 3600 + 60, 11 * 3600 + 60, 12 * 3600],
        ),
    )
    return builder.build()


def test_direct_runs_finds_the_train_and_its_calling_points(timetable):
    runs = direct_runs(timetable, {"740000001"}, {"740000003"}, DAY)
    assert len(runs) == 1
    run = runs[0]
    assert [p.stop.name for p in run.points] == [
        "Stockholm C", "Nässjö C", "Alvesta", "Malmö C"
    ]
    assert run.departure.hour == 7
    assert run.arrival.hour == 12


def test_a_non_sj_operator_is_not_scanned(timetable):
    assert direct_runs(timetable, {"740000001"}, {"740000003"}, DAY, operators=frozenset()) == []


def test_the_wrong_direction_is_not_a_direct_run(timetable):
    assert direct_runs(timetable, {"740000003"}, {"740000001"}, DAY) == []


def test_an_intermediate_stop_can_be_the_destination(timetable):
    runs = direct_runs(timetable, {"740000001"}, {"740000004"}, DAY)
    assert len(runs) == 1
    assert [p.stop.name for p in runs[0].points] == ["Stockholm C", "Nässjö C", "Alvesta"]
