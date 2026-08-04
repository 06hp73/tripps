"""Helpers for building small, exactly-known timetables in tests."""

from __future__ import annotations

from tripps.models import Stop, TransportMode
from tripps.routing.timetable import RouteInfo, Timetable, TimetableBuilder, Trip

# Roughly-real coordinates so segment distances (and therefore price floors) are sane.
COORDS: dict[str, tuple[float, float]] = {
    "STO": (59.3300, 18.0590),  # Stockholm C
    "NRK": (58.5960, 16.1830),  # Norrkoping
    "LIN": (58.4160, 15.6250),  # Linkoping
    "GBG": (57.7089, 11.9746),  # Goteborg
    "MMX": (55.6090, 13.0000),  # Malmo
    "BLE": (60.4845, 15.4379),  # Borlange
    "ARN": (59.6519, 17.9186),  # Arlanda
}


def hhmm(hours: int, minutes: int = 0) -> int:
    """Service-day seconds. Values >= 24h are legal (GTFS after-midnight departures)."""
    return hours * 3600 + minutes * 60


def stop(code: str, modes: frozenset[TransportMode] = frozenset()) -> Stop:
    lat, lon = COORDS[code]
    return Stop(id=code, name=code, lat=lat, lon=lon, modes=modes)


class Net:
    """Fluent builder: `Net().route(...).transfer(...).build()`."""

    def __init__(self) -> None:
        self.b = TimetableBuilder()

    def stops(self, *codes: str) -> Net:
        for code in codes:
            self.b.add_stop(stop(code))
        return self

    def route(
        self,
        route_id: str,
        codes: list[str],
        trips: list[list[tuple[int, int]]],
        *,
        mode: TransportMode = TransportMode.TRAIN,
        operator: str | None = "TEST",
        fares_ore: list[int | None] | None = None,
        synthetic: bool = False,
    ) -> Net:
        """`trips[t][i]` is the (arrival, departure) pair at `codes[i]` for trip t."""
        self.stops(*codes)
        info = RouteInfo(id=route_id, mode=mode, operator=operator, synthetic=synthetic)
        fares = fares_ore or [None] * len(trips)
        for t, times in enumerate(trips):
            self.b.add_trip(
                info,
                codes,
                Trip(
                    id=f"{route_id}#{t}",
                    arrivals=[a for a, _ in times],
                    departures=[d for _, d in times],
                    precomputed_fare_ore=fares[t],
                ),
            )
        return self

    def transfer(self, a: str, b: str, seconds: int) -> Net:
        self.b.add_transfer(a, b, seconds)
        return self

    def build(self) -> Timetable:
        return self.b.build()


def at(t: int) -> tuple[int, int]:
    """A stop where arrival == departure (no dwell)."""
    return (t, t)


# --- reference McRAPTOR (pre-optimization), for frontier-equality tests --------------------
#
# Verbatim algorithmic copy of the straightforward implementation that predates the
# tuple-bag + cumulative-floor + target-potential engineering. The optimized router must
# produce the IDENTICAL (arrival, price, departure) frontier; these ~120 lines are the
# oracle that proves it and must not be "improved".

from tripps.routing.mcraptor import Label, RideEdge, TransferEdge  # noqa: E402
from tripps.routing.timetable import INFINITY  # noqa: E402


def _ref_dominates(a_arr, a_price, a_dep, b_arr, b_price, b_dep):
    return a_arr <= b_arr and a_price <= b_price and a_dep >= b_dep


def _ref_merge(bag, label):
    for existing in bag:
        if _ref_dominates(existing.arrival, existing.price_ore, existing.departure,
                          label.arrival, label.price_ore, label.departure):
            return False
    bag[:] = [k for k in bag
              if not _ref_dominates(label.arrival, label.price_ore, label.departure,
                                    k.arrival, k.price_ore, k.departure)]
    bag.append(label)
    return True


def _ref_boardable(route, pos, ready, query, unboarded, fares_vary):
    first = route.earliest_trip_from(pos, ready)
    if first is None:
        return []
    trips = route.trips

    def ok(i):
        nb = trips[i].no_board
        return nb is None or not nb[pos]

    if not fares_vary and not (unboarded and query.profile):
        for i in range(first, len(trips)):
            if ok(i):
                return [i]
        return []
    horizon = ready + query.profile_window_seconds
    within = []
    for i in range(first, len(trips)):
        if trips[i].departures[pos] > horizon:
            break
        if ok(i):
            within.append(i)
    cap = (
        query.max_departures_per_route_intercity
        if route.info.mode in (TransportMode.TRAIN, TransportMode.BUS)
        else query.max_departures_per_route
    )
    if fares_vary or len(within) <= cap:
        return within
    n = len(within)
    picks = sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})
    return [within[i] for i in picks]


def _ref_relax(tt, curr_bag, best_bag, marked, query, target_bag):
    if TransportMode.WALK not in query.allowed_modes:
        return
    newly = set()
    for stop_idx in list(marked):
        for label in list(curr_bag[stop_idx]):
            if isinstance(label.edge, TransferEdge):
                continue
            for to_stop, seconds in tt.transfers[stop_idx]:
                arrival = label.arrival + seconds
                if arrival > query.latest_arrival:
                    continue
                if any(_ref_dominates(x.arrival, x.price_ore, x.departure,
                                      arrival, label.price_ore, label.departure)
                       for x in target_bag):
                    continue
                walked = Label(stop=to_stop, arrival=arrival, price_ore=label.price_ore,
                               departure=label.departure, parent=label,
                               edge=TransferEdge(from_stop=stop_idx, seconds=seconds))
                if _ref_merge(best_bag[to_stop], walked):
                    _ref_merge(curr_bag[to_stop], walked)
                    newly.add(to_stop)
                    if to_stop in query.targets:
                        _ref_merge(target_bag, walked)
    marked |= newly


def run_mcraptor_reference(tt, floors, query):
    """Frontier oracle: returns the set of (arrival, price_ore, departure) at the targets."""
    n = tt.num_stops
    prev_bag = [[] for _ in range(n)]
    best_bag = [[] for _ in range(n)]
    target_bag = []

    marked = set()
    for stop_idx, depart_at in query.origins:
        label = Label(stop=stop_idx, arrival=depart_at, price_ore=0)
        if _ref_merge(prev_bag[stop_idx], label):
            _ref_merge(best_bag[stop_idx], label)
            marked.add(stop_idx)

    _ref_relax(tt, prev_bag, best_bag, marked, query, target_bag)
    for stop_idx in query.targets:
        for lbl in best_bag[stop_idx]:
            _ref_merge(target_bag, lbl)

    for _round in range(1, query.max_rounds + 1):
        if not marked:
            break
        curr_bag = [[] for _ in range(n)]
        route_entry = {}
        for stop_idx in marked:
            for route_idx, pos in tt.stop_routes[stop_idx]:
                if tt.routes[route_idx].info.mode not in query.allowed_modes:
                    continue
                cur = route_entry.get(route_idx)
                if cur is None or pos < cur:
                    route_entry[route_idx] = pos
        marked = set()

        for route_idx, first_pos in route_entry.items():
            route = tt.routes[route_idx]
            per_km = floors.per_km_ore(route.info.mode, route.info.operator)
            base_fare = floors.boarding_ore(route.info.mode, route.info.operator)
            fares_vary = len({t.precomputed_fare_ore for t in route.trips}) > 1
            rides = []  # [trip_idx, price, board_pos, departure, parent]

            for pos in range(first_pos, len(route.stops)):
                stop_idx = route.stops[pos]
                if rides and pos > first_pos:
                    inc = int(per_km * route.segment_km[pos - 1])
                    for r in rides:
                        r[1] += inc
                for r in rides:
                    trip = route.trips[r[0]]
                    if trip.no_alight is not None and trip.no_alight[pos]:
                        continue
                    arrival = trip.arrivals[pos]
                    if arrival > query.latest_arrival:
                        continue
                    if any(_ref_dominates(x.arrival, x.price_ore, x.departure,
                                          arrival, r[1], r[3]) for x in target_bag):
                        continue
                    label = Label(stop=stop_idx, arrival=arrival, price_ore=r[1],
                                  departure=r[3], parent=r[4],
                                  edge=RideEdge(route_idx, r[0], r[2], pos))
                    if _ref_merge(best_bag[stop_idx], label):
                        _ref_merge(curr_bag[stop_idx], label)
                        marked.add(stop_idx)
                        if stop_idx in query.targets:
                            _ref_merge(target_bag, label)
                for label in prev_bag[stop_idx]:
                    ready = label.ready_at(query.min_transfer_seconds)
                    unboarded = label.departure == INFINITY
                    for trip_idx in _ref_boardable(route, pos, ready, query, unboarded, fares_vary):
                        trip = route.trips[trip_idx]
                        pf = trip.precomputed_fare_ore
                        fare = pf if pf is not None else base_fare
                        price = label.price_ore + fare
                        departure = trip.departures[pos] if unboarded else label.departure
                        if not any(r[0] <= trip_idx and r[1] <= price and r[3] >= departure
                                   for r in rides):
                            rides = [r for r in rides
                                     if not (trip_idx <= r[0] and price <= r[1]
                                             and departure >= r[3])]
                            rides.append([trip_idx, price, pos, departure, label])

        _ref_relax(tt, curr_bag, best_bag, marked, query, target_bag)
        prev_bag = curr_bag

    return {(x.arrival, x.price_ore, x.departure) for x in target_bag}
