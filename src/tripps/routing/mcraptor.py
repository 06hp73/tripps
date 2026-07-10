"""Multi-criteria RAPTOR (McRAPTOR) over (arrival time, price floor).

Why a custom core rather than OpenTripPlanner 2, MOTIS or R5: none of them optimizes a
dynamically-priced ticket *during* the search. OTP2 computes fares from finished
itineraries; MOTIS/nigiri attaches fares after routing with compile-time-fixed criteria;
R5's in-routing fare McRAPTOR ships only static rule-based regional calculators, which
cannot express a yield-managed SJ fare. Optimizing price only after routing means the
cheapest-but-slower itinerary is often never generated to be priced.

So the router optimizes a *lower bound* on price (see `floors.py`) alongside arrival
time, and phase 2 replaces bounds with real quotes. The bound's correctness contract is
the load-bearing invariant of this whole design:

    floor(leg) <= true_price(leg),  always.

If it ever exceeds the truth, McRAPTOR can prune the genuinely cheapest journey before
anyone tries to price it. `db.record_reprice_delta` logs every (floor, actual) pair so
violations are detectable rather than silent.

Rounds carry the third criterion: a label found in round k used at most k vehicles, so
transfers need not enter the Pareto comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..models import TransportMode
from .floors import PriceFloorModel
from .timetable import INFINITY, Timetable


@dataclass(frozen=True, slots=True)
class RideEdge:
    """Provenance: this label was reached by riding a vehicle."""

    route_idx: int
    trip_idx: int
    board_pos: int
    alight_pos: int


@dataclass(frozen=True, slots=True)
class TransferEdge:
    """Provenance: this label was reached on foot."""

    from_stop: int
    seconds: int


@dataclass(frozen=True, slots=True)
class Label:
    """A Pareto-optimal way to be at `stop` at `arrival`, having left at `departure`.

    `departure` is the moment the journey first boards a vehicle, and `INFINITY` until it
    does (an unboarded traveller is free to leave whenever, which is the best possible
    value of a criterion we maximize).
    """

    stop: int
    arrival: int
    price_ore: int
    departure: int = INFINITY
    parent: Label | None = None
    edge: RideEdge | TransferEdge | None = None

    @property
    def arrived_by_vehicle(self) -> bool:
        return isinstance(self.edge, RideEdge)

    def ready_at(self, min_transfer_seconds: int) -> int:
        """Earliest second this label can board a vehicle at its stop.

        A passenger who just got off a train needs the minimum change time. A passenger
        who walked here has already spent that time walking, and one who started here
        needs nothing.
        """
        return self.arrival + (min_transfer_seconds if self.arrived_by_vehicle else 0)


@dataclass(slots=True)
class _Ride:
    """A label currently on board, used only inside one route scan."""

    trip_idx: int
    price_ore: int
    board_pos: int
    departure: int
    parent: Label


def _dominates(a_arr: int, a_price: int, a_dep: int, b_arr: int, b_price: int, b_dep: int) -> bool:
    """Profile dominance: no later, no dearer, and leaves no earlier.

    The third criterion is what makes this a range (profile) query rather than a single
    earliest-arrival search, and it is not optional for a planner that claims to find the
    cheapest trip.

    The price a leg *actually* costs varies by departure - a FlixBus from Stockholm to
    Goteborg is 800 SEK at 07:30 and 420 SEK at 22:55 on the same day - but the routing
    price floor is derived from distance and operator, so it is identical for both. Without
    a departure criterion the night bus is "later arrival, same price" and gets pruned
    before phase 2 ever prices it. Keeping later-departing journeys on the frontier is the
    only way the cheapest one survives to be priced.
    """
    return a_arr <= b_arr and a_price <= b_price and a_dep >= b_dep


def _merge(bag: list[Label], label: Label, *, max_size: int | None = None) -> bool:
    """Insert `label` if non-dominated; drop everything it dominates. True if inserted."""
    for existing in bag:
        if _dominates(
            existing.arrival, existing.price_ore, existing.departure,
            label.arrival, label.price_ore, label.departure,
        ):
            return False
    bag[:] = [
        keep
        for keep in bag
        if not _dominates(
            label.arrival, label.price_ore, label.departure,
            keep.arrival, keep.price_ore, keep.departure,
        )
    ]
    bag.append(label)
    if max_size is not None and len(bag) > max_size:
        # Bounded mode is approximate by construction: keep the extremes of the frontier
        # (fastest and cheapest) and thin the middle.
        bag.sort(key=lambda x: (x.arrival, x.price_ore))
        keep_head = bag[: max_size // 2]
        keep_tail = sorted(bag, key=lambda x: (x.price_ore, -x.departure))[
            : max_size - len(keep_head)
        ]
        merged = {id(x): x for x in (*keep_head, *keep_tail)}
        bag[:] = list(merged.values())
    return True


def _dominated_by_any(bag: list[Label], arrival: int, price_ore: int, departure: int) -> bool:
    """Target pruning.

    Extending a journey can only push its arrival later and its price higher, and never
    changes when it left. So a label that some finished journey already dominates can be
    discarded outright.
    """
    return any(
        _dominates(x.arrival, x.price_ore, x.departure, arrival, price_ore, departure)
        for x in bag
    )


@dataclass(slots=True)
class RaptorResult:
    """Pareto-optimal labels at the target, plus diagnostics."""

    labels: list[Label]
    rounds_run: int
    bag_capped: bool = False
    stats: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class RaptorQuery:
    #: (stop index, earliest departure second at that stop)
    origins: list[tuple[int, int]]
    targets: set[int]
    max_rounds: int = 6
    min_transfer_seconds: int = 300
    allowed_modes: frozenset[TransportMode] = frozenset(TransportMode)
    #: None keeps the search exact. A cap trades correctness for speed.
    max_bag_size: int | None = None
    #: Ignore journeys arriving after this second (e.g. user's latest_arrival).
    latest_arrival: int = INFINITY
    #: Range (profile) query: keep later-departing journeys on the frontier so that a
    #: cheaper-but-later departure is not pruned before it can be priced.
    profile: bool = True
    #: How far past the requested departure a profile query looks for departures. A full day,
    #: because a 22:55 night coach is routinely the cheapest way across Sweden and a shorter
    #: window silently excludes it from ever being priced.
    profile_window_seconds: int = 24 * 3600
    #: Cap on origin departures enumerated per route, so a metro line with a train every
    #: three minutes does not swamp the frontier.
    max_departures_per_route: int = 16


def run_mcraptor(
    tt: Timetable, floors: PriceFloorModel, query: RaptorQuery
) -> RaptorResult:
    """Exact multi-criteria RAPTOR. Returns the Pareto frontier over the target stops."""
    n = tt.num_stops
    # bags[k][stop]: labels using exactly k vehicle boardings.
    prev_bag: list[list[Label]] = [[] for _ in range(n)]
    best_bag: list[list[Label]] = [[] for _ in range(n)]
    target_bag: list[Label] = []
    capped = False
    stats = {"route_scans": 0, "boardings": 0, "labels_created": 0}

    marked: set[int] = set()
    for stop_idx, depart_at in query.origins:
        label = Label(stop=stop_idx, arrival=depart_at, price_ore=0)
        if _merge(prev_bag[stop_idx], label, max_size=query.max_bag_size):
            _merge(best_bag[stop_idx], label, max_size=query.max_bag_size)
            marked.add(stop_idx)

    # Origin footpaths: reaching a nearby stop on foot costs no boarding.
    _relax_transfers(tt, prev_bag, best_bag, marked, query, target_bag)
    for stop_idx in query.targets:
        for lbl in best_bag[stop_idx]:
            _merge(target_bag, lbl, max_size=query.max_bag_size)

    rounds_run = 0
    for _round in range(1, query.max_rounds + 1):
        if not marked:
            break
        rounds_run = _round
        curr_bag: list[list[Label]] = [[] for _ in range(n)]

        # Collect routes touched by marked stops, remembering the earliest boardable position.
        route_entry: dict[int, int] = {}
        for stop_idx in marked:
            for route_idx, pos in tt.stop_routes[stop_idx]:
                if tt.routes[route_idx].info.mode not in query.allowed_modes:
                    continue
                current = route_entry.get(route_idx)
                if current is None or pos < current:
                    route_entry[route_idx] = pos
        marked = set()

        for route_idx, first_pos in route_entry.items():
            stats["route_scans"] += 1
            route = tt.routes[route_idx]
            per_km = floors.per_km_ore(route.info.mode, route.info.operator)
            rides: list[_Ride] = []

            for pos in range(first_pos, len(route.stops)):
                stop_idx = route.stops[pos]

                if rides and pos > first_pos:
                    increment = int(per_km * route.segment_km[pos - 1])
                    for ride in rides:
                        ride.price_ore += increment

                # Alight: every on-board label offers an arrival at this stop.
                for ride in rides:
                    trip = route.trips[ride.trip_idx]
                    arrival = trip.arrivals[pos]
                    if arrival > query.latest_arrival:
                        continue
                    if _dominated_by_any(target_bag, arrival, ride.price_ore, ride.departure):
                        continue  # target pruning: remaining legs only add time and cost
                    label = Label(
                        stop=stop_idx,
                        arrival=arrival,
                        price_ore=ride.price_ore,
                        departure=ride.departure,
                        parent=ride.parent,
                        edge=RideEdge(route_idx, ride.trip_idx, ride.board_pos, pos),
                    )
                    if _merge(best_bag[stop_idx], label, max_size=query.max_bag_size):
                        _merge(curr_bag[stop_idx], label, max_size=query.max_bag_size)
                        marked.add(stop_idx)
                        stats["labels_created"] += 1
                        if stop_idx in query.targets:
                            _merge(target_bag, label, max_size=query.max_bag_size)

                # Board: labels from the previous round may catch a trip here.
                for label in prev_bag[stop_idx]:
                    ready = label.ready_at(query.min_transfer_seconds)
                    unboarded = label.departure == INFINITY
                    for trip_idx in _boardable_trips(route, pos, ready, query, unboarded):
                        trip = route.trips[trip_idx]
                        boarding_fare = (
                            trip.precomputed_fare_ore
                            if trip.precomputed_fare_ore is not None
                            else floors.boarding_ore(route.info.mode, route.info.operator)
                        )
                        price = label.price_ore + boarding_fare
                        # The journey's departure is fixed by its first boarding.
                        departure = (
                            trip.departures[pos]
                            if label.departure == INFINITY
                            else label.departure
                        )
                        new_ride = _Ride(
                            trip_idx=trip_idx,
                            price_ore=price,
                            board_pos=pos,
                            departure=departure,
                            parent=label,
                        )
                        if not _dominates_existing_ride(rides, new_ride):
                            rides = _insert_ride(rides, new_ride)
                            stats["boardings"] += 1

            if query.max_bag_size is not None and any(
                len(b) >= query.max_bag_size for b in curr_bag
            ):
                capped = True

        _relax_transfers(tt, curr_bag, best_bag, marked, query, target_bag)
        prev_bag = curr_bag

    return RaptorResult(
        labels=sorted(target_bag, key=lambda x: (x.price_ore, x.arrival)),
        rounds_run=rounds_run,
        bag_capped=capped,
        stats=stats,
    )


def _boardable_trips(
    route, pos: int, ready: int, query: RaptorQuery, unboarded: bool
) -> list[int]:
    """Which trips on this route are worth boarding at `pos`, given readiness at `ready`.

    Mid-journey, only the earliest catchable trip can ever help: a later one on the same
    route arrives later, costs the same boarding fare, and cannot change a departure time
    that is already fixed, so it is dominated outright.

    At the origin, the opposite holds. The journey's departure is whatever trip it boards,
    and leaving later is a criterion we maximize, so every subsequent departure is on the
    frontier. Enumerating them is what turns this into a range query and what lets a cheap
    late-night coach be found alongside the fast morning one. The window and the cap keep
    that enumeration from running to the end of the service day on a busy urban route.
    """
    first = route.earliest_trip_from(pos, ready)
    if first is None:
        return []
    if not (unboarded and query.profile):
        return [first]

    horizon = ready + query.profile_window_seconds
    chosen: list[int] = []
    for trip_idx in range(first, len(route.trips)):
        if route.trips[trip_idx].departures[pos] > horizon:
            break
        chosen.append(trip_idx)
        if len(chosen) >= query.max_departures_per_route:
            break
    return chosen


def _dominates_existing_ride(rides: list[_Ride], candidate: _Ride) -> bool:
    """An earlier, no dearer, no earlier-departing ride makes `candidate` pointless.

    Trips are index-ordered by departure and validated non-overtaking, so a lower trip_idx
    arrives no later at every remaining stop.
    """
    return any(
        r.trip_idx <= candidate.trip_idx
        and r.price_ore <= candidate.price_ore
        and r.departure >= candidate.departure
        for r in rides
    )


def _insert_ride(rides: list[_Ride], candidate: _Ride) -> list[_Ride]:
    kept = [
        r
        for r in rides
        if not (
            candidate.trip_idx <= r.trip_idx
            and candidate.price_ore <= r.price_ore
            and candidate.departure >= r.departure
        )
    ]
    kept.append(candidate)
    return kept


def _relax_transfers(
    tt: Timetable,
    curr_bag: list[list[Label]],
    best_bag: list[list[Label]],
    marked: set[int],
    query: RaptorQuery,
    target_bag: list[Label],
) -> None:
    """Walk from every stop marked in this round. Footpaths are transitively closed,
    so one relaxation pass suffices and walking never chains into another walk."""
    if TransportMode.WALK not in query.allowed_modes:
        return
    newly_marked: set[int] = set()
    for stop_idx in list(marked):
        for label in list(curr_bag[stop_idx]):
            if isinstance(label.edge, TransferEdge):
                continue  # no walk-after-walk; the closure already covers it
            for to_stop, seconds in tt.transfers[stop_idx]:
                arrival = label.arrival + seconds
                if arrival > query.latest_arrival:
                    continue
                if _dominated_by_any(target_bag, arrival, label.price_ore, label.departure):
                    continue
                walked = Label(
                    stop=to_stop,
                    arrival=arrival,
                    price_ore=label.price_ore,
                    departure=label.departure,
                    parent=label,
                    edge=TransferEdge(from_stop=stop_idx, seconds=seconds),
                )
                if _merge(best_bag[to_stop], walked, max_size=query.max_bag_size):
                    _merge(curr_bag[to_stop], walked, max_size=query.max_bag_size)
                    newly_marked.add(to_stop)
                    if to_stop in query.targets:
                        _merge(target_bag, walked, max_size=query.max_bag_size)
    marked |= newly_marked


def unwind(label: Label) -> list[Label]:
    """Origin-first chain of labels that produced `label`."""
    chain: list[Label] = []
    node: Label | None = label
    while node is not None:
        chain.append(node)
        node = node.parent
    return list(reversed(chain))


def relabel_prices(label: Label, new_prices: list[int]) -> Label:
    """Rebuild a label chain with corrected cumulative prices (used after phase-2 repricing)."""
    chain = unwind(label)
    if len(new_prices) != len(chain):
        raise ValueError(f"expected {len(chain)} prices, got {len(new_prices)}")
    rebuilt: Label | None = None
    for node, price in zip(chain, new_prices, strict=True):
        rebuilt = replace(node, price_ore=price, parent=rebuilt)
    assert rebuilt is not None
    return rebuilt
