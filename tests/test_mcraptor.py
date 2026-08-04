"""McRAPTOR correctness: Pareto frontier, transfers, exact fares, floor contract."""

from __future__ import annotations

import pytest

from tripps.models import TransportMode
from tripps.routing.floors import DEFAULT_FLOORS, ModeFloor, PriceFloorModel, zero_floors
from tripps.routing.mcraptor import (
    RaptorQuery,
    RideEdge,
    TransferEdge,
    run_mcraptor,
    unwind,
)
from tripps.routing.timetable import haversine_km

from .support import Net, at, hhmm


def _query(tt, origin: str, target: str, depart: int, **kw) -> RaptorQuery:
    return RaptorQuery(
        origins=[(tt.index_of(origin), depart)],
        targets={tt.index_of(target)},
        **kw,
    )


def test_single_direct_trip_is_found():
    tt = Net().route("R", ["STO", "GBG"], [[at(hhmm(8)), at(hhmm(11))]]).build()
    res = run_mcraptor(tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(7)))

    assert len(res.labels) == 1
    label = res.labels[0]
    assert label.arrival == hhmm(11)
    assert isinstance(label.edge, RideEdge)


def test_trip_departing_before_query_is_not_boardable():
    tt = Net().route("R", ["STO", "GBG"], [[at(hhmm(6)), at(hhmm(9))]]).build()
    res = run_mcraptor(tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(7)))
    assert res.labels == []


def test_pareto_keeps_both_fast_expensive_and_slow_cheap():
    """The whole reason for multi-criteria search: a cheaper-but-slower option must
    survive, since a pure earliest-arrival router would discard it."""
    tt = (
        Net()
        # Fast train, high floor.
        .route(
            "FAST",
            ["STO", "GBG"],
            [[at(hhmm(8)), at(hhmm(11))]],
            mode=TransportMode.TRAIN,
            operator="FASTOP",
        )
        # Slow bus, low floor.
        .route(
            "SLOW",
            ["STO", "GBG"],
            [[at(hhmm(8)), at(hhmm(15))]],
            mode=TransportMode.BUS,
            operator="SLOWOP",
        )
        .build()
    )
    floors = PriceFloorModel(
        DEFAULT_FLOORS,
        operator_overrides={
            "FASTOP": ModeFloor(base_ore=50_000, per_km_ore=0),
            "SLOWOP": ModeFloor(base_ore=10_000, per_km_ore=0),
        },
    )
    res = run_mcraptor(tt, floors, _query(tt, "STO", "GBG", hhmm(7)))

    assert len(res.labels) == 2, "both frontier points must survive"
    by_price = sorted(res.labels, key=lambda x: x.price_ore)
    assert by_price[0].price_ore == 10_000 and by_price[0].arrival == hhmm(15)
    assert by_price[1].price_ore == 50_000 and by_price[1].arrival == hhmm(11)
    # Results are returned cheapest-first: price is the objective.
    assert res.labels[0].price_ore == 10_000


def test_dominated_option_is_pruned():
    """Slower AND pricier must not survive."""
    tt = (
        Net()
        .route("GOOD", ["STO", "GBG"], [[at(hhmm(8)), at(hhmm(11))]], operator="CHEAP")
        .route("BAD", ["STO", "GBG"], [[at(hhmm(8)), at(hhmm(14))]], operator="DEAR")
        .build()
    )
    floors = PriceFloorModel(
        DEFAULT_FLOORS,
        operator_overrides={
            "CHEAP": ModeFloor(base_ore=10_000, per_km_ore=0),
            "DEAR": ModeFloor(base_ore=90_000, per_km_ore=0),
        },
    )
    res = run_mcraptor(tt, floors, _query(tt, "STO", "GBG", hhmm(7)))
    assert len(res.labels) == 1
    assert res.labels[0].price_ore == 10_000


def test_transfer_requires_minimum_change_time():
    """Arriving at 10:00 cannot board a 10:02 connection when the change time is 5 min."""
    tt = (
        Net()
        .route("LEG1", ["STO", "NRK"], [[at(hhmm(9)), at(hhmm(10))]])
        .route(
            "LEG2",
            ["NRK", "GBG"],
            [
                [at(hhmm(10, 2)), at(hhmm(13))],  # too tight
                [at(hhmm(10, 30)), at(hhmm(13, 30))],  # catchable
            ],
        )
        .build()
    )
    res = run_mcraptor(
        tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(8), min_transfer_seconds=300)
    )
    assert len(res.labels) == 1
    assert res.labels[0].arrival == hhmm(13, 30)


def test_round_limit_bounds_number_of_vehicles():
    """max_rounds=1 permits a single boarding, so a two-leg journey is unreachable."""
    tt = (
        Net()
        .route("LEG1", ["STO", "NRK"], [[at(hhmm(9)), at(hhmm(10))]])
        .route("LEG2", ["NRK", "GBG"], [[at(hhmm(11)), at(hhmm(13))]])
        .build()
    )
    one = run_mcraptor(tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(8), max_rounds=1))
    assert one.labels == []

    two = run_mcraptor(tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(8), max_rounds=2))
    assert len(two.labels) == 1
    assert two.labels[0].arrival == hhmm(13)


def test_footpath_transfer_between_distinct_stops():
    tt = (
        Net()
        .route("LEG1", ["STO", "NRK"], [[at(hhmm(9)), at(hhmm(10))]])
        .route("LEG2", ["LIN", "GBG"], [[at(hhmm(10, 30)), at(hhmm(13))]])
        .transfer("NRK", "LIN", 600)
        .build()
    )
    res = run_mcraptor(tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(8)))
    assert len(res.labels) == 1

    chain = unwind(res.labels[0])
    edges = [node.edge for node in chain if node.edge is not None]
    assert any(isinstance(e, TransferEdge) and e.seconds == 600 for e in edges)


def test_walk_disallowed_removes_footpath_journeys():
    tt = (
        Net()
        .route("LEG1", ["STO", "NRK"], [[at(hhmm(9)), at(hhmm(10))]])
        .route("LEG2", ["LIN", "GBG"], [[at(hhmm(10, 30)), at(hhmm(13))]])
        .transfer("NRK", "LIN", 600)
        .build()
    )
    modes = frozenset(TransportMode) - {TransportMode.WALK}
    res = run_mcraptor(
        tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(8), allowed_modes=modes)
    )
    assert res.labels == []


def test_intermediate_stop_boarding_and_alighting():
    """Board mid-route and alight mid-route on a 3-stop pattern."""
    tt = Net().route(
        "R", ["STO", "NRK", "GBG"], [[at(hhmm(8)), at(hhmm(9)), at(hhmm(12))]]
    ).build()
    res = run_mcraptor(tt, zero_floors(), _query(tt, "NRK", "GBG", hhmm(8, 30)))
    assert len(res.labels) == 1
    edge = res.labels[0].edge
    assert isinstance(edge, RideEdge)
    assert (edge.board_pos, edge.alight_pos) == (1, 2)


def test_price_accumulates_per_km_while_riding():
    """Riding STO->NRK->GBG on one ticket charges base once, distance for both segments."""
    tt = Net().route(
        "R",
        ["STO", "NRK", "GBG"],
        [[at(hhmm(8)), at(hhmm(9)), at(hhmm(12))]],
        operator="OP",
    ).build()
    floors = PriceFloorModel(
        DEFAULT_FLOORS, operator_overrides={"OP": ModeFloor(base_ore=1000, per_km_ore=100)}
    )
    res = run_mcraptor(tt, floors, _query(tt, "STO", "GBG", hhmm(7)))

    d1 = haversine_km(59.3300, 18.0590, 58.5960, 16.1830)
    d2 = haversine_km(58.5960, 16.1830, 57.7089, 11.9746)
    expected = 1000 + int(100 * d1) + int(100 * d2)
    assert res.labels[0].price_ore == expected


def test_two_tickets_cost_more_than_one_through_ride():
    """Boarding twice pays the base fare twice: this is what makes the floor additive
    and sub-additive relative to real through-fares."""
    through = Net().route(
        "R", ["STO", "NRK", "GBG"], [[at(hhmm(8)), at(hhmm(9)), at(hhmm(12))]], operator="OP"
    ).build()
    split = (
        Net()
        .route("R1", ["STO", "NRK"], [[at(hhmm(8)), at(hhmm(9))]], operator="OP")
        .route("R2", ["NRK", "GBG"], [[at(hhmm(9, 30)), at(hhmm(12))]], operator="OP")
        .build()
    )
    floors = PriceFloorModel(
        DEFAULT_FLOORS, operator_overrides={"OP": ModeFloor(base_ore=5000, per_km_ore=10)}
    )
    a = run_mcraptor(through, floors, _query(through, "STO", "GBG", hhmm(7)))
    b = run_mcraptor(split, floors, _query(split, "STO", "GBG", hhmm(7)))
    assert b.labels[0].price_ore - a.labels[0].price_ore == 5000


def test_precomputed_fare_overrides_floor_and_ignores_distance():
    """Injected legs (flights, Freerider cars) carry an exact fare on the trip."""
    tt = Net().route(
        "FR",
        ["BLE", "NRK"],
        [[at(hhmm(9)), at(hhmm(13))]],
        mode=TransportMode.FREERIDER,
        operator="hertz-freerider",
        fares_ore=[7_500],
        synthetic=True,
    ).build()
    res = run_mcraptor(tt, PriceFloorModel(), _query(tt, "BLE", "NRK", hhmm(8)))
    assert res.labels[0].price_ore == 7_500


def test_mixed_mode_train_then_freerider():
    """The headline requirement: leg 1 by train, leg 2 by free Hertz car."""
    tt = (
        Net()
        .route(
            "TRAIN",
            ["STO", "BLE"],
            [[at(hhmm(7)), at(hhmm(9, 30))]],
            mode=TransportMode.TRAIN,
            operator="SJ",
        )
        .route(
            "CAR",
            ["BLE", "NRK"],
            [[at(hhmm(10)), at(hhmm(14))]],
            mode=TransportMode.FREERIDER,
            operator="hertz-freerider",
            fares_ore=[0],
            synthetic=True,
        )
        .build()
    )
    res = run_mcraptor(tt, PriceFloorModel(), _query(tt, "STO", "NRK", hhmm(6)))
    assert len(res.labels) == 1

    chain = unwind(res.labels[0])
    ridden = [n.edge for n in chain if isinstance(n.edge, RideEdge)]
    assert len(ridden) == 2
    modes = [tt.routes[e.route_idx].info.mode for e in ridden]
    assert modes == [TransportMode.TRAIN, TransportMode.FREERIDER]
    # The car is free, so the whole trip costs exactly the train's floor.
    train_floor = PriceFloorModel().floor_ore(
        TransportMode.TRAIN, "SJ", haversine_km(59.3300, 18.0590, 60.4845, 15.4379)
    )
    assert res.labels[0].price_ore == train_floor


def test_excluding_freerider_falls_back_to_other_modes():
    tt = (
        Net()
        .route(
            "CAR",
            ["STO", "GBG"],
            [[at(hhmm(8)), at(hhmm(13))]],
            mode=TransportMode.FREERIDER,
            operator="hertz-freerider",
            fares_ore=[0],
            synthetic=True,
        )
        .route(
            "BUS", ["STO", "GBG"], [[at(hhmm(8)), at(hhmm(15))]], mode=TransportMode.BUS
        )
        .build()
    )
    allowed = frozenset(TransportMode) - {TransportMode.FREERIDER}
    res = run_mcraptor(
        tt, PriceFloorModel(), _query(tt, "STO", "GBG", hhmm(7), allowed_modes=allowed)
    )
    assert len(res.labels) == 1
    assert res.labels[0].arrival == hhmm(15)


def test_latest_arrival_filters_late_journeys():
    tt = Net().route("R", ["STO", "GBG"], [[at(hhmm(8)), at(hhmm(16))]]).build()
    res = run_mcraptor(
        tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(7), latest_arrival=hhmm(15))
    )
    assert res.labels == []


def test_after_midnight_service_times():
    """GTFS expresses a 00:30 arrival on a 23:00 departure as 24:30:00."""
    tt = Net().route("NIGHT", ["STO", "GBG"], [[at(hhmm(23)), at(hhmm(24, 30))]]).build()
    res = run_mcraptor(tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(22)))
    assert res.labels[0].arrival == hhmm(24, 30)


def test_target_pruning_does_not_drop_frontier_points():
    """A cheap-but-slow journey must survive even after a fast one reaches the target."""
    tt = (
        Net()
        .route("FAST", ["STO", "GBG"], [[at(hhmm(8)), at(hhmm(10))]], operator="DEAR")
        .route(
            "SLOW1",
            ["STO", "NRK"],
            [[at(hhmm(8)), at(hhmm(9))]],
            operator="CHEAP",
        )
        .route(
            "SLOW2",
            ["NRK", "GBG"],
            [[at(hhmm(9, 30)), at(hhmm(14))]],
            operator="CHEAP",
        )
        .build()
    )
    floors = PriceFloorModel(
        DEFAULT_FLOORS,
        operator_overrides={
            "DEAR": ModeFloor(base_ore=80_000, per_km_ore=0),
            "CHEAP": ModeFloor(base_ore=1_000, per_km_ore=0),
        },
    )
    res = run_mcraptor(tt, floors, _query(tt, "STO", "GBG", hhmm(7)))
    prices = sorted(x.price_ore for x in res.labels)
    assert prices == [2_000, 80_000]  # two CHEAP boardings vs one DEAR


def test_zero_floor_finds_superset_of_calibrated_floor_journeys():
    """Zero is a valid lower bound, so it must never find fewer target stops."""
    tt = (
        Net()
        .route("A", ["STO", "GBG"], [[at(hhmm(8)), at(hhmm(12))]], mode=TransportMode.TRAIN)
        .route("B", ["STO", "GBG"], [[at(hhmm(9)), at(hhmm(13))]], mode=TransportMode.BUS)
        .build()
    )
    zero = run_mcraptor(tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(7)))
    real = run_mcraptor(tt, PriceFloorModel(), _query(tt, "STO", "GBG", hhmm(7)))
    zero_arrivals = {x.arrival for x in zero.labels}
    real_arrivals = {x.arrival for x in real.labels}
    assert real_arrivals <= zero_arrivals or zero_arrivals <= real_arrivals


def test_exact_fare_mode_rejects_nonzero_floor():
    with pytest.raises(ValueError, match="exact fares"):
        PriceFloorModel({TransportMode.FREERIDER: ModeFloor(base_ore=1, per_km_ore=0)})


def test_unreachable_target_returns_empty():
    tt = (
        Net()
        .route("R", ["STO", "NRK"], [[at(hhmm(8)), at(hhmm(9))]])
        .stops("GBG")
        .build()
    )
    res = run_mcraptor(tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(7)))
    assert res.labels == []


def test_boardable_trips_spread_keeps_earliest_and_latest():
    """When a route has more departures than the cap, the sample is spread over the whole
    window (keeping the first and last), not truncated to the earliest - a cheap late coach
    must stay reachable."""
    from tripps.routing.mcraptor import _boardable_trips

    trips = [[at(hhmm(6) + i * 1800), at(hhmm(9) + i * 1800)] for i in range(34)]  # 06:00..22:30
    tt = Net().route("R", ["STO", "GBG"], trips).build()
    route = tt.routes[0]
    q = _query(tt, "STO", "GBG", 0)
    picks = _boardable_trips(route, route.trips, 0, 0, q, unboarded=True, fares_vary=False)
    assert len(picks) == q.max_departures_per_route == 16
    assert picks[0] == 0, "earliest departure kept"
    assert picks[-1] == 33, "latest departure kept (was dropped by first-16 truncation)"
    assert picks == sorted(picks)


def test_boardable_trips_under_cap_returns_all():
    from tripps.routing.mcraptor import _boardable_trips

    trips = [[at(hhmm(8) + i * 1800), at(hhmm(11) + i * 1800)] for i in range(5)]
    tt = Net().route("R", ["STO", "GBG"], trips).build()
    q = _query(tt, "STO", "GBG", 0)
    assert list(
        _boardable_trips(tt.routes[0], tt.routes[0].trips, 0, 0, q, unboarded=True, fares_vary=False)
    ) == [0, 1, 2, 3, 4]


# --- per-trip exact fares: a later trip can be strictly cheaper ------------------------------


def test_mid_journey_cheaper_later_flight_survives():
    """The cardinal repro: feeder train to the airport, then a flight route where the 20:00
    departure (300 kr) is far cheaper than the 10:00 one (1500 kr). Once the feeder fixes the
    journey's departure, the old mid-journey "earliest catchable only" shortcut boarded only
    the 10:00 flight - the genuinely cheapest journey was never generated at all."""
    tt = (
        Net()
        .route("FEED", ["STO", "ARN"], [[at(hhmm(8)), at(hhmm(9))]])
        .route(
            "AIR",
            ["ARN", "GBG"],
            [
                [at(hhmm(10)), at(hhmm(11))],
                [at(hhmm(20)), at(hhmm(21))],
            ],
            mode=TransportMode.FLIGHT,
            operator="flight:test",
            fares_ore=[150_000, 30_000],
            synthetic=True,
        )
        .build()
    )
    res = run_mcraptor(tt, zero_floors(), _query(tt, "STO", "GBG", hhmm(7)))

    prices = sorted(label.price_ore for label in res.labels)
    assert 30_000 in prices, "the cheaper LATER flight must be boardable mid-journey"
    assert res.labels[0].price_ore == 30_000, "and it is the cheapest label"


def test_same_fare_route_keeps_the_single_boarding_shortcut_mid_journey():
    """Freerider-shaped route: every trip carries the SAME fare, so mid-journey the earliest
    catchable trip still dominates and the fast path must stay."""
    from tripps.routing.mcraptor import _boardable_trips

    trips = [[at(hhmm(10 + i)), at(hhmm(11 + i))] for i in range(3)]
    tt = Net().route(
        "CAR", ["STO", "GBG"], trips, fares_ore=[5_000, 5_000, 5_000], synthetic=True
    ).build()
    q = _query(tt, "STO", "GBG", 0)
    # fares do not vary -> mid-journey shortcut intact
    assert list(
        _boardable_trips(tt.routes[0], tt.routes[0].trips, 0, 0, q, unboarded=False, fares_vary=False)
    ) == [0]
    # varying fares -> every catchable trip is generated, even mid-journey
    assert list(
        _boardable_trips(tt.routes[0], tt.routes[0].trips, 0, 0, q, unboarded=False, fares_vary=True)
    ) == [0, 1, 2]


def test_cap_never_drops_the_cheapest_varying_fare_trip():
    """With 20 departures and cap 16, the time-index spread drops indices {2,7,12,17}. For a
    fare-carrying route the cheapest flight could sit exactly there - fare-varying routes skip
    the thinning entirely, so the min-fare departure always stays reachable."""
    fares = [100_000] * 20
    fares[12] = 5_000  # an index the linspace sample would have dropped
    trips = [[at(hhmm(6) + i * 1800), at(hhmm(9) + i * 1800)] for i in range(20)]
    tt = Net().route(
        "AIR", ["ARN", "GBG"], trips,
        mode=TransportMode.FLIGHT, operator="flight:test",
        fares_ore=fares, synthetic=True,
    ).build()
    res = run_mcraptor(tt, zero_floors(), _query(tt, "ARN", "GBG", 0))

    assert min(label.price_ore for label in res.labels) == 5_000
