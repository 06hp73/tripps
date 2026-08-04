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

Two engineering layers keep the exact search fast (both proven frontier-identical to the
straightforward implementation by tests/support.py's reference copy):

- Bags hold `(arrival, price, departure, ready, Label)` tuples so dominance compares
  local ints, and the frozen Label is only constructed once an entry actually survives
  domination. Per-route cumulative floor arrays replace the per-stop ride increment (the
  same `int(per_km * segment)` per segment, so prices are bit-identical).
- A*-style target potentials: a backward Dijkstra from the targets over the static
  min-segment-time graph yields `rem[stop]`, a true lower bound on the remaining seconds
  to any target (waiting, minimum change times, and boarding restrictions are ignored,
  which only ever UNDERestimates - the safe side). A label is discarded when even its
  optimistic completion `(arrival + rem, price, departure)` is dominated by a finished
  journey: extensions only arrive later and cost more, so the real completion would be
  dominated too. Same argument as plain target pruning, just with a tighter clock.
"""

from __future__ import annotations

import heapq
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


# Profile dominance: no later, no dearer, and leaves no earlier. The departure criterion
# is what makes this a range (profile) query - a FlixBus is 800 SEK at 07:30 and 420 SEK
# at 22:55 with the SAME distance-derived floor, so without it the night bus would be
# "later arrival, same price" and be pruned before phase 2 ever prices it. In the hot
# loops the comparison is inlined over bag-entry ints; this named form is for reference
# and for the cold paths.
def _dominates(a_arr: int, a_price: int, a_dep: int, b_arr: int, b_price: int, b_dep: int) -> bool:
    return a_arr <= b_arr and a_price <= b_price and a_dep >= b_dep


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
    #: A*-style pruning against a per-stop lower bound on remaining travel time. Provably
    #: frontier-preserving (see module docstring); the switch exists for debugging only.
    target_potentials: bool = True


def target_potentials(tt: Timetable, query: RaptorQuery) -> list[int]:
    """`rem[stop]`: lower bound on seconds from `stop` to the nearest target.

    Backward Dijkstra over the static graph whose edge weights are the minimum
    in-vehicle time of any trip over each route segment, plus footpaths. Waiting time,
    minimum change time, and no_board/no_alight restrictions are deliberately ignored:
    each omission only lowers the bound, and only a bound that is never above the truth
    may prune. INFINITY means no target is reachable at all under the allowed modes.
    """
    n = tt.num_stops
    radj: list[list[tuple[int, int]]] = [[] for _ in range(n)]  # to -> [(from, seconds)]
    for route in tt.routes:
        if route.info.mode not in query.allowed_modes:
            continue
        stops = route.stops
        trips = route.trips
        for i in range(len(stops) - 1):
            w = min(t.arrivals[i + 1] - t.departures[i] for t in trips)
            if w < 0:
                w = 0
            radj[stops[i + 1]].append((stops[i], w))
    if TransportMode.WALK in query.allowed_modes:
        for a in range(n):
            for b, seconds in tt.transfers[a]:
                radj[b].append((a, seconds))

    rem = [INFINITY] * n
    pq: list[tuple[int, int]] = []
    for t in query.targets:
        rem[t] = 0
        pq.append((0, t))
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > rem[u]:
            continue
        for v, w in radj[u]:
            nd = d + w
            if nd < rem[v]:
                rem[v] = nd
                heapq.heappush(pq, (nd, v))
    return rem


def _thin(bag: list[tuple], max_size: int) -> None:
    """Bounded-mode thinning, same policy as always: keep the extremes of the frontier
    (fastest and cheapest ends) and drop the middle. Approximate by construction."""
    if len(bag) <= max_size:
        return
    head = sorted(bag, key=lambda e: (e[0], e[1]))[: max_size // 2]
    tail = sorted(bag, key=lambda e: (e[1], -e[2]))[: max_size - len(head)]
    merged = {id(e[4]): e for e in (*head, *tail)}
    bag[:] = list(merged.values())


def run_mcraptor(
    tt: Timetable, floors: PriceFloorModel, query: RaptorQuery
) -> RaptorResult:
    """Exact multi-criteria RAPTOR. Returns the Pareto frontier over the target stops."""
    n = tt.num_stops
    mts = query.min_transfer_seconds
    latest = query.latest_arrival
    max_size = query.max_bag_size
    targets = query.targets
    capped = False
    stats = {"route_scans": 0, "boardings": 0, "labels_created": 0}

    rem = (
        target_potentials(tt, query)
        if query.target_potentials
        else [0] * n  # a zero potential degenerates to plain target pruning
    )

    # Bags hold (arrival, price, departure, ready, Label); dominance compares the ints.
    prev_bag: list[list[tuple]] = [[] for _ in range(n)]
    best_bag: list[list[tuple]] = [[] for _ in range(n)]
    target_bag: list[tuple] = []
    # Scalar guards over target_bag: a candidate beating any of these minima/maximum
    # cannot be dominated, so the linear scan is skipped for most probes.
    tguard = [INFINITY, INFINITY, -1]  # min arrival, min price, max departure

    def bag_insert(bag: list[tuple], arr: int, price: int, dep: int, ready: int, label) -> bool:
        for e in bag:
            if e[0] <= arr and e[1] <= price and e[2] >= dep:
                return False
        bag[:] = [e for e in bag if not (arr <= e[0] and price <= e[1] and dep >= e[2])]
        bag.append((arr, price, dep, ready, label))
        if max_size is not None:
            _thin(bag, max_size)
        return True

    def target_insert(arr: int, price: int, dep: int, ready: int, label) -> None:
        if bag_insert(target_bag, arr, price, dep, ready, label):
            if arr < tguard[0]:
                tguard[0] = arr
            if price < tguard[1]:
                tguard[1] = price
            if dep > tguard[2]:
                tguard[2] = dep

    def future_dead(stop_idx: int, arr: int, price: int, dep: int) -> bool:
        """Target pruning with the potential's tighter clock.

        Any completion of this label arrives at `arr + rem[stop]` at the earliest, costs
        at least `price`, and departs exactly at `dep` - so if a finished journey already
        dominates that OPTIMISTIC completion, it dominates every real one.
        """
        opt = arr + rem[stop_idx]
        if opt > latest:
            return True  # also catches rem == INFINITY: the target is unreachable
        if opt < tguard[0] or price < tguard[1] or dep > tguard[2]:
            return False
        for e in target_bag:
            if e[0] <= opt and e[1] <= price and e[2] >= dep:
                return True
        return False

    allowed = [r.info.mode in query.allowed_modes for r in tt.routes]
    cum_cache: dict[int, list[int]] = {}
    fares_vary_cache: dict[int, bool] = {}

    marked: set[int] = set()
    for stop_idx, depart_at in query.origins:
        if future_dead(stop_idx, depart_at, 0, INFINITY):
            continue  # target unreachable from here (or past latest_arrival already)
        label = Label(stop=stop_idx, arrival=depart_at, price_ore=0)
        if bag_insert(prev_bag[stop_idx], depart_at, 0, INFINITY, depart_at, label):
            bag_insert(best_bag[stop_idx], depart_at, 0, INFINITY, depart_at, label)
            marked.add(stop_idx)

    # Origin footpaths: reaching a nearby stop on foot costs no boarding.
    _relax_transfers(tt, prev_bag, best_bag, marked, query, target_insert, future_dead, bag_insert)
    for stop_idx in targets:
        for e in best_bag[stop_idx]:
            target_insert(*e)

    rounds_run = 0
    for _round in range(1, query.max_rounds + 1):
        if not marked:
            break
        rounds_run = _round
        curr_bag: list[list[tuple]] = [[] for _ in range(n)]

        # Collect routes touched by marked stops, remembering the earliest boardable position.
        route_entry: dict[int, int] = {}
        for stop_idx in marked:
            for route_idx, pos in tt.stop_routes[stop_idx]:
                if not allowed[route_idx]:
                    continue
                current = route_entry.get(route_idx)
                if current is None or pos < current:
                    route_entry[route_idx] = pos
        marked = set()

        for route_idx, first_pos in route_entry.items():
            stats["route_scans"] += 1
            route = tt.routes[route_idx]
            trips = route.trips
            rstops = route.stops

            cum = cum_cache.get(route_idx)
            if cum is None:
                # Prefix sums of the per-segment floor increments - the same
                # int(per_km * segment) rounding per segment as the incremental form,
                # so prices are bit-identical while rides stay O(1) per stop.
                per_km = floors.per_km_ore(route.info.mode, route.info.operator)
                acc = 0
                cum = [0]
                for seg in route.segment_km:
                    acc += int(per_km * seg)
                    cum.append(acc)
                cum_cache[route_idx] = cum
            base_fare = floors.boarding_ore(route.info.mode, route.info.operator)

            # Do this route's trips carry differing exact fares (one flight per departure)?
            # If so, a LATER trip can be strictly cheaper, and the usual "only the earliest
            # catchable trip matters" boarding shortcut is unsound (see _boardable_trips).
            fares_vary = fares_vary_cache.get(route_idx)
            if fares_vary is None:
                fares_vary = len({t.precomputed_fare_ore for t in trips}) > 1
                fares_vary_cache[route_idx] = fares_vary

            # Rides as (trip_idx, base_price, board_pos, departure, parent_label), where
            # base_price = price_at(pos) - cum[pos]; constant along the ride, so ride
            # dominance on it is equivalent to dominance on the running price.
            rides: list[tuple] = []

            for pos in range(first_pos, len(rstops)):
                stop_idx = rstops[pos]
                cpos = cum[pos]

                # Alight: every on-board label offers an arrival at this stop.
                for ride in rides:
                    trip_idx, ride_base, board_pos, ride_dep, ride_parent = ride
                    trip = trips[trip_idx]
                    no_alight = trip.no_alight
                    if no_alight is not None and no_alight[pos]:
                        continue  # set-down forbidden here (GTFS drop_off_type != 0)
                    arrival = trip.arrivals[pos]
                    if arrival > latest:
                        continue
                    price = ride_base + cpos
                    if future_dead(stop_idx, arrival, price, ride_dep):
                        continue
                    bb = best_bag[stop_idx]
                    dominated = False
                    for e in bb:
                        if e[0] <= arrival and e[1] <= price and e[2] >= ride_dep:
                            dominated = True
                            break
                    if dominated:
                        continue
                    # Survived: only now pay for the frozen Label construction.
                    label = Label(
                        stop=stop_idx,
                        arrival=arrival,
                        price_ore=price,
                        departure=ride_dep,
                        parent=ride_parent,
                        edge=RideEdge(route_idx, trip_idx, board_pos, pos),
                    )
                    ready = arrival + mts
                    bb[:] = [
                        e for e in bb
                        if not (arrival <= e[0] and price <= e[1] and ride_dep >= e[2])
                    ]
                    bb.append((arrival, price, ride_dep, ready, label))
                    if max_size is not None:
                        _thin(bb, max_size)
                    bag_insert(curr_bag[stop_idx], arrival, price, ride_dep, ready, label)
                    marked.add(stop_idx)
                    stats["labels_created"] += 1
                    if stop_idx in targets:
                        target_insert(arrival, price, ride_dep, ready, label)

                # Board: labels from the previous round may catch a trip here.
                for entry in prev_bag[stop_idx]:
                    _larr, lprice, ldep, ready, llabel = entry
                    unboarded = ldep == INFINITY
                    for trip_idx in _boardable_trips(
                        route, trips, pos, ready, query, unboarded, fares_vary
                    ):
                        pf = trips[trip_idx].precomputed_fare_ore
                        boarding_fare = pf if pf is not None else base_fare
                        price = lprice + boarding_fare
                        # The journey's departure is fixed by its first boarding.
                        departure = trips[trip_idx].departures[pos] if unboarded else ldep
                        base = price - cpos
                        # An earlier, no dearer, no earlier-departing ride makes this one
                        # pointless (trips are sorted and validated non-overtaking).
                        pointless = False
                        for r in rides:
                            if r[0] <= trip_idx and r[1] <= base and r[3] >= departure:
                                pointless = True
                                break
                        if not pointless:
                            rides = [
                                r for r in rides
                                if not (trip_idx <= r[0] and base <= r[1] and departure >= r[3])
                            ]
                            rides.append((trip_idx, base, pos, departure, llabel))
                            stats["boardings"] += 1

            if max_size is not None and any(len(b) >= max_size for b in curr_bag):
                capped = True

        _relax_transfers(
            tt, curr_bag, best_bag, marked, query, target_insert, future_dead, bag_insert
        )
        prev_bag = curr_bag

    return RaptorResult(
        labels=sorted((e[4] for e in target_bag), key=lambda x: (x.price_ore, x.arrival)),
        rounds_run=rounds_run,
        bag_capped=capped,
        stats=stats,
    )


def _boardable_trips(
    route, trips, pos: int, ready: int, query: RaptorQuery, unboarded: bool, fares_vary: bool
) -> list[int] | tuple[int, ...]:
    """Which trips on this route are worth boarding at `pos`, given readiness at `ready`.

    Mid-journey, only the earliest catchable trip can ever help: a later one on the same
    route arrives later, costs the same boarding fare, and cannot change a departure time
    that is already fixed, so it is dominated outright.

    At the origin, the opposite holds. The journey's departure is whatever trip it boards,
    and leaving later is a criterion we maximize, so every subsequent departure is on the
    frontier. Enumerating them is what turns this into a range query and what lets a cheap
    late-night coach be found alongside the fast morning one. The window and the cap keep
    that enumeration from running to the end of the service day on a busy urban route.

    `fares_vary` breaks BOTH shortcuts. When each trip carries its own exact fare (one
    flight per departure), a later trip can be strictly CHEAPER, so the mid-journey
    "earliest only" rule would prune the cheapest journey the moment a feeder leg precedes
    the flight, and the time-index thinning could sample away the one cheap departure.
    Such routes enumerate every catchable trip in the window - they are synthetic and
    small (domestic flights: at most a couple of dozen departures per airport pair per
    day), so the full enumeration is bounded. The ride-level Pareto keeps only the
    non-dominated ones.
    """
    # Inlined binary search over departures at `pos` (route.earliest_trip_from, without
    # the method-call overhead: this runs once per (label, stop) probe).
    lo, hi = 0, len(trips)
    while lo < hi:
        mid = (lo + hi) // 2
        if trips[mid].departures[pos] < ready:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(trips):
        return ()
    first = lo

    if not fares_vary and not (unboarded and query.profile):
        # The earliest catchable trip may be set-down-only here (pickup_type != 0); the
        # earliest trip that actually SELLS a boarding at this stop is the dominating one.
        for trip_idx in range(first, len(trips)):
            no_board = trips[trip_idx].no_board
            if no_board is None or not no_board[pos]:
                return (trip_idx,)
        return ()

    horizon = ready + query.profile_window_seconds
    within: list[int] = []
    for trip_idx in range(first, len(trips)):
        if trips[trip_idx].departures[pos] > horizon:
            break
        no_board = trips[trip_idx].no_board
        if no_board is None or not no_board[pos]:
            within.append(trip_idx)
    cap = query.max_departures_per_route
    if fares_vary or len(within) <= cap:
        return within
    # More departures than the cap. Taking the first `cap` in order would keep only the
    # earliest ones and silently drop the whole evening - and a cheap 22:55 coach is routinely
    # the bargain. Spread the sample evenly over the window instead, keeping BOTH the earliest
    # and the latest departure (linspace over [0, n-1] including both ends).
    n = len(within)
    picks = sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})
    return [within[i] for i in picks]


def _relax_transfers(
    tt: Timetable,
    curr_bag: list[list[tuple]],
    best_bag: list[list[tuple]],
    marked: set[int],
    query: RaptorQuery,
    target_insert,
    future_dead,
    bag_insert,
) -> None:
    """Walk from every stop marked in this round. Footpaths are transitively closed,
    so one relaxation pass suffices and walking never chains into another walk."""
    if TransportMode.WALK not in query.allowed_modes:
        return
    latest = query.latest_arrival
    max_size = query.max_bag_size
    newly_marked: set[int] = set()
    for stop_idx in list(marked):
        for entry in list(curr_bag[stop_idx]):
            arr, price, dep, _ready, label = entry
            if isinstance(label.edge, TransferEdge):
                continue  # no walk-after-walk; the closure already covers it
            for to_stop, seconds in tt.transfers[stop_idx]:
                w_arr = arr + seconds
                if w_arr > latest:
                    continue
                if future_dead(to_stop, w_arr, price, dep):
                    continue
                bb = best_bag[to_stop]
                dominated = False
                for e in bb:
                    if e[0] <= w_arr and e[1] <= price and e[2] >= dep:
                        dominated = True
                        break
                if dominated:
                    continue
                walked = Label(
                    stop=to_stop,
                    arrival=w_arr,
                    price_ore=price,
                    departure=dep,
                    parent=label,
                    edge=TransferEdge(from_stop=stop_idx, seconds=seconds),
                )
                bb[:] = [
                    e for e in bb if not (w_arr <= e[0] and price <= e[1] and dep >= e[2])
                ]
                bb.append((w_arr, price, dep, w_arr, walked))
                if max_size is not None:
                    _thin(bb, max_size)
                bag_insert(curr_bag[to_stop], w_arr, price, dep, w_arr, walked)
                newly_marked.add(to_stop)
                if to_stop in query.targets:
                    target_insert(w_arr, price, dep, w_arr, walked)
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
