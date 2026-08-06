"""The optimized router's frontier must be IDENTICAL to the straightforward reference.

Every engineering layer in mcraptor.py (tuple bags, lazy labels, cumulative floors, A*
target potentials) claims to be semantics-preserving; this file is the proof. Any diff in
the (arrival, price, departure) frontier on any of these networks is a correctness bug,
never an acceptable optimization artifact.
"""

from __future__ import annotations

from tripps.models import TransportMode
from tripps.routing.floors import DEFAULT_FLOORS, ModeFloor, PriceFloorModel, zero_floors
from tripps.routing.mcraptor import RaptorQuery, run_mcraptor
from tripps.routing.timetable import Trip

from .support import Net, at, hhmm, run_mcraptor_reference


def _frontier(res) -> set[tuple[int, int, int]]:
    return {(x.arrival, x.price_ore, x.departure) for x in res.labels}


def _assert_equal(tt, floors, query) -> None:
    got = _frontier(run_mcraptor(tt, floors, query))
    want = run_mcraptor_reference(tt, floors, query)
    assert got == want, f"frontier mismatch:\n new-only: {got - want}\n ref-only: {want - got}"
    assert got, "degenerate check: these fixtures must actually produce journeys"


def _q(tt, origin, target, depart, **kw):
    return RaptorQuery(origins=[(tt.index_of(origin), depart)],
                       targets={tt.index_of(target)}, **kw)


def _profile_net():
    """Two competing routes with many departures + a mid-journey transfer + footpath."""
    net = Net()
    net.route("EXPRESS", ["STO", "NRK", "GBG"],
              [[at(hhmm(6 + i)), at(hhmm(7 + i)), at(hhmm(9 + i))] for i in range(14)],
              operator="FAST")
    net.route("COACH", ["STO", "GBG"],
              [[at(hhmm(7, 30 * i)), at(hhmm(14, 30 * i))] for i in range(2)],
              mode=TransportMode.BUS, operator="SLOW")
    net.route("LOCAL", ["LIN", "GBG"],
              [[at(hhmm(10 + i)), at(hhmm(11 + i))] for i in range(8)],
              operator="LOC")
    net.transfer("NRK", "LIN", 600)
    return net.build()


def _floors():
    return PriceFloorModel(DEFAULT_FLOORS, operator_overrides={
        "FAST": ModeFloor(base_ore=40_000, per_km_ore=10),
        "SLOW": ModeFloor(base_ore=9_000, per_km_ore=2),
        "LOC": ModeFloor(base_ore=3_000, per_km_ore=5),
    })


def test_equal_on_profile_net_with_transfers():
    tt = _profile_net()
    _assert_equal(tt, _floors(), _q(tt, "STO", "GBG", hhmm(6)))


def test_equal_with_latest_arrival_cut():
    tt = _profile_net()
    _assert_equal(tt, _floors(), _q(tt, "STO", "GBG", hhmm(6), latest_arrival=hhmm(13)))


def test_equal_with_varying_fares_flight_route():
    net = Net()
    net.route("FEED", ["STO", "ARN"], [[at(hhmm(7 + i)), at(hhmm(8 + i))] for i in range(4)])
    net.route("AIR", ["ARN", "GBG"],
              [[at(hhmm(10)), at(hhmm(11))], [at(hhmm(15)), at(hhmm(16))],
               [at(hhmm(20)), at(hhmm(21))]],
              mode=TransportMode.FLIGHT, operator="flight:x", synthetic=True,
              fares_ore=[150_000, 90_000, 30_000])
    tt = net.build()
    _assert_equal(tt, zero_floors(), _q(tt, "STO", "GBG", hhmm(6)))


def test_equal_with_boarding_restrictions():
    net = Net()
    net.route("R", ["STO", "NRK", "LIN", "GBG"],
              [[at(hhmm(8 + i)), at(hhmm(9 + i)), at(hhmm(10 + i)), at(hhmm(12 + i))]
               for i in range(5)])
    tt = net.build()
    # Forbid boarding at NRK and alighting at LIN on every trip, after the fact.
    for route in tt.routes:
        route.trips = [
            Trip(id=t.id, arrivals=t.arrivals, departures=t.departures, headsign=t.headsign,
                 precomputed_fare_ore=t.precomputed_fare_ore,
                 no_board=(False, True, False, False), no_alight=(False, False, True, False))
            for t in route.trips
        ]
    _assert_equal(tt, _floors(), _q(tt, "STO", "GBG", hhmm(6)))
    q2 = _q(tt, "NRK", "GBG", hhmm(6))
    assert _frontier(run_mcraptor(tt, _floors(), q2)) == run_mcraptor_reference(tt, _floors(), q2) == set()


def test_equal_multi_origin_and_origin_walk():
    tt = _profile_net()
    q = RaptorQuery(
        origins=[(tt.index_of("STO"), hhmm(6)), (tt.index_of("NRK"), hhmm(6, 30))],
        targets={tt.index_of("GBG")},
    )
    _assert_equal(tt, _floors(), q)


def test_equal_when_origin_is_also_target():
    tt = _profile_net()
    q = RaptorQuery(origins=[(tt.index_of("STO"), hhmm(6))],
                    targets={tt.index_of("STO"), tt.index_of("GBG")})
    got = _frontier(run_mcraptor(tt, _floors(), q))
    want = run_mcraptor_reference(tt, _floors(), q)
    assert got == want


def test_equal_with_walk_disallowed():
    tt = _profile_net()
    modes = frozenset(TransportMode) - {TransportMode.WALK}
    _assert_equal(tt, _floors(), _q(tt, "STO", "GBG", hhmm(6), allowed_modes=modes))


def test_equal_with_departure_cap_engaged():
    net = Net()
    net.route("METRO", ["STO", "GBG"],
              [[at(hhmm(6) + i * 900), at(hhmm(9) + i * 900)] for i in range(40)],
              operator="M")
    tt = net.build()
    _assert_equal(tt, _floors(), _q(tt, "STO", "GBG", 0))


def test_potentials_off_matches_potentials_on():
    """The debug switch must not change results either."""
    tt = _profile_net()
    on = run_mcraptor(tt, _floors(), _q(tt, "STO", "GBG", hhmm(6)))
    off = run_mcraptor(tt, _floors(), _q(tt, "STO", "GBG", hhmm(6), target_potentials=False))
    assert _frontier(on) == _frontier(off)
