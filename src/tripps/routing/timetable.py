"""RAPTOR-native timetable structures.

Deliberately not a graph. RAPTOR's own contribution is that a public-transit network
is better represented as arrays of routes and trips than as a time-expanded or
time-dependent graph; at intercity scale (hundreds of stops, low thousands of
departures/day) each round is a linear scan and no priority queue is needed.

A "route" here is the RAPTOR sense of the word, not the GTFS sense: the set of trips
that visit *exactly* the same stop sequence. One GTFS route_id with two stopping
patterns becomes two RAPTOR routes.

Times are integer seconds from the start of the service day (03:00 local, by GTFS
convention departures after midnight are 24:00:00+). Distances are precomputed per
segment so the price floor can accumulate additively while riding.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..models import Stop, TransportMode

#: Sentinel "unreachable" time. Larger than any plausible service-day second.
INFINITY = 2**31 - 1

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance. Used for segment lengths and as the road-matrix fallback."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass(frozen=True, slots=True)
class RouteInfo:
    """Immutable descriptor of one RAPTOR route."""

    id: str
    mode: TransportMode
    operator: str | None
    #: GTFS extended route_type, or a synthetic sentinel for injected routes.
    route_type: int = -1
    #: True for routes synthesized per-search (Freerider offers, flights), which are
    #: never part of the persisted GTFS timetable.
    synthetic: bool = False


@dataclass(slots=True)
class Trip:
    """One vehicle run along a route's stop sequence.

    `arrivals[i]` / `departures[i]` align with `Route.stops[i]`.
    `precomputed_fare_ore` is set for injected routes whose price is already known
    exactly (a flight offer, a Freerider car); for GTFS routes it stays None and the
    price floor model supplies a bound instead.
    """

    id: str
    arrivals: list[int]
    departures: list[int]
    headsign: str | None = None
    precomputed_fare_ore: int | None = None

    def __post_init__(self) -> None:
        if len(self.arrivals) != len(self.departures):
            raise ValueError(f"trip {self.id}: arrivals/departures length mismatch")


@dataclass(slots=True)
class Route:
    """A stop sequence plus the trips that serve it, sorted by departure time."""

    info: RouteInfo
    stops: list[int]
    trips: list[Trip]
    #: segment_km[i] = distance from stops[i] to stops[i+1]; length len(stops) - 1.
    segment_km: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.stops) < 2:
            raise ValueError(f"route {self.info.id}: needs >= 2 stops")

    def sort_trips(self) -> None:
        """Order trips by departure from the first stop.

        RAPTOR's route scan assumes trips do not overtake one another; a later-departing
        trip must not arrive earlier at any stop. Real intercity data violates this
        (an express overtakes a stopping service on the same pattern) only when the two
        share an identical stop sequence, which by construction of RAPTOR routes means
        they are genuinely comparable. We sort by first departure and validate.
        """
        self.trips.sort(key=lambda t: t.departures[0])

    def overtaking_trips(self) -> list[tuple[str, str]]:
        """Pairs (earlier_id, later_id) that violate the no-overtaking assumption."""
        bad: list[tuple[str, str]] = []
        for a, b in zip(self.trips, self.trips[1:], strict=False):
            if any(x > y for x, y in zip(a.arrivals, b.arrivals, strict=False)):
                bad.append((a.id, b.id))
        return bad

    def earliest_trip_from(self, stop_pos: int, after: int) -> int | None:
        """Index of the first trip departing `stops[stop_pos]` at or after `after`.

        Binary search over trips, which are sorted and (validated) non-overtaking.
        """
        lo, hi = 0, len(self.trips)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.trips[mid].departures[stop_pos] < after:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < len(self.trips) else None


@dataclass(slots=True)
class Timetable:
    """The immutable network the router scans."""

    stops: list[Stop]
    routes: list[Route]
    #: stop index -> [(route index, position of stop within that route)]
    stop_routes: list[list[tuple[int, int]]]
    #: stop index -> [(to stop index, seconds)] transitively closed footpaths
    transfers: list[list[tuple[int, int]]]
    stop_index: dict[str, int]

    @property
    def num_stops(self) -> int:
        return len(self.stops)

    @property
    def num_trips(self) -> int:
        return sum(len(r.trips) for r in self.routes)

    def index_of(self, stop_id: str) -> int:
        try:
            return self.stop_index[stop_id]
        except KeyError as exc:
            raise KeyError(f"unknown stop id: {stop_id}") from exc

    def validate(self) -> list[str]:
        """Structural self-check. Returns human-readable problems, empty when sound."""
        problems: list[str] = []
        for r_idx, route in enumerate(self.routes):
            if len(route.segment_km) != len(route.stops) - 1:
                problems.append(
                    f"route {route.info.id}: segment_km has {len(route.segment_km)}, "
                    f"expected {len(route.stops) - 1}"
                )
            for trip in route.trips:
                if len(trip.arrivals) != len(route.stops):
                    problems.append(
                        f"route {route.info.id} trip {trip.id}: "
                        f"{len(trip.arrivals)} stop_times for {len(route.stops)} stops"
                    )
                    continue
                for i in range(len(trip.arrivals) - 1):
                    if trip.departures[i] > trip.arrivals[i + 1]:
                        problems.append(
                            f"route {route.info.id} trip {trip.id}: time travel at stop {i}"
                        )
                        break
            for earlier, later in route.overtaking_trips():
                problems.append(f"route {route.info.id}: trip {later} overtakes {earlier}")
            for pos, s_idx in enumerate(route.stops):
                if (r_idx, pos) not in self.stop_routes[s_idx]:
                    problems.append(
                        f"route {route.info.id} pos {pos}: missing from stop_routes"
                    )
        return problems


class TimetableBuilder:
    """Accumulates stops/patterns, then materializes a `Timetable`.

    Patterns are keyed by (route info identity, stop-sequence tuple) so that a single
    GTFS route with several stopping patterns yields several RAPTOR routes, as required.
    """

    def __init__(self) -> None:
        self._stops: list[Stop] = []
        self._stop_index: dict[str, int] = {}
        self._patterns: dict[tuple, tuple[RouteInfo, list[int], list[Trip]]] = {}
        self._transfers: dict[int, dict[int, int]] = defaultdict(dict)

    def add_stop(self, stop: Stop) -> int:
        """Idempotent by stop id. Returns the stop's index."""
        existing = self._stop_index.get(stop.id)
        if existing is not None:
            return existing
        idx = len(self._stops)
        self._stops.append(stop)
        self._stop_index[stop.id] = idx
        return idx

    def has_stop(self, stop_id: str) -> bool:
        return stop_id in self._stop_index

    def get_stop(self, stop_id: str) -> Stop:
        return self._stops[self._stop_index[stop_id]]

    def index_of(self, stop_id: str) -> int:
        return self._stop_index[stop_id]

    def stop_at(self, index: int) -> Stop:
        return self._stops[index]

    def add_trip(self, info: RouteInfo, stop_ids: list[str], trip: Trip) -> None:
        if len(stop_ids) != len(trip.arrivals):
            raise ValueError(
                f"trip {trip.id}: {len(stop_ids)} stops vs {len(trip.arrivals)} stop_times"
            )
        indices = [self._stop_index[s] for s in stop_ids]
        key = (info.id, info.mode, info.operator, info.route_type, tuple(indices))
        if key not in self._patterns:
            self._patterns[key] = (info, indices, [])
        self._patterns[key][2].append(trip)

    def add_transfer(self, from_id: str, to_id: str, seconds: int, *, bidirectional: bool = True) -> None:
        a, b = self._stop_index[from_id], self._stop_index[to_id]
        if a == b:
            return
        # Keep the fastest footpath when several are offered for the same pair.
        if b not in self._transfers[a] or self._transfers[a][b] > seconds:
            self._transfers[a][b] = seconds
        if bidirectional and (a not in self._transfers[b] or self._transfers[b][a] > seconds):
            self._transfers[b][a] = seconds

    def build(self, *, drop_overtaking: bool = True) -> Timetable:
        """Materialize. Sorts trips, computes segment distances, indexes stop->routes.

        `drop_overtaking` removes the later trip of any overtaking pair. RAPTOR's route
        scan would otherwise silently return non-optimal journeys; dropping is the
        conservative choice (we lose a trip rather than return a wrong answer), and the
        count is reported by the caller via `Timetable.validate`.
        """
        routes: list[Route] = []
        for info, stop_indices, trips in self._patterns.values():
            route = Route(info=info, stops=stop_indices, trips=trips)
            route.sort_trips()
            if drop_overtaking:
                route.trips = _strip_overtaking(route.trips)
            route.segment_km = [
                haversine_km(
                    self._stops[a].lat, self._stops[a].lon, self._stops[b].lat, self._stops[b].lon
                )
                for a, b in zip(stop_indices, stop_indices[1:], strict=False)
            ]
            routes.append(route)

        stop_routes: list[list[tuple[int, int]]] = [[] for _ in self._stops]
        for r_idx, route in enumerate(routes):
            for pos, s_idx in enumerate(route.stops):
                stop_routes[s_idx].append((r_idx, pos))

        transfers: list[list[tuple[int, int]]] = [[] for _ in self._stops]
        for a, targets in self._transfers.items():
            transfers[a] = sorted(targets.items())

        return Timetable(
            stops=list(self._stops),
            routes=routes,
            stop_routes=stop_routes,
            transfers=transfers,
            stop_index=dict(self._stop_index),
        )


@dataclass(slots=True)
class RouteAddition:
    """A synthetic route to overlay onto a prebuilt timetable."""

    info: RouteInfo
    stops: list[Stop]
    trips: list[Trip]


def overlay_timetable(
    base: Timetable,
    additions: list[RouteAddition],
    transfers: list[tuple[str, str, int]] | None = None,
) -> Timetable:
    """Return `base` plus synthetic routes, without reparsing the GTFS feed.

    Freerider cars and flight offers are query-scoped: their inventory changes every few
    minutes and depends on the origin/destination being searched, so they cannot live in
    the daily-built timetable. Rebuilding the whole timetable per request would mean
    re-reading the feed; instead the stop table and route index are copied (cheap: a few
    hundred thousand integers at intercity scale) and the new routes appended.

    Existing `Route` objects are shared, not copied. They are treated as immutable once
    built; nothing in the router mutates a route.
    """
    stops = list(base.stops)
    stop_index = dict(base.stop_index)
    routes = list(base.routes)
    stop_routes = [list(entries) for entries in base.stop_routes]
    transfer_lists = [list(entries) for entries in base.transfers]

    def ensure_stop(stop: Stop) -> int:
        existing = stop_index.get(stop.id)
        if existing is not None:
            return existing
        idx = len(stops)
        stops.append(stop)
        stop_index[stop.id] = idx
        stop_routes.append([])
        transfer_lists.append([])
        return idx

    for addition in additions:
        if len(addition.stops) < 2:
            raise ValueError(f"synthetic route {addition.info.id}: needs >= 2 stops")
        indices = [ensure_stop(s) for s in addition.stops]
        for trip in addition.trips:
            if len(trip.arrivals) != len(indices):
                raise ValueError(
                    f"synthetic route {addition.info.id} trip {trip.id}: "
                    f"{len(trip.arrivals)} stop_times for {len(indices)} stops"
                )
        route = Route(info=addition.info, stops=indices, trips=list(addition.trips))
        route.sort_trips()
        route.trips = _strip_overtaking(route.trips)
        route.segment_km = [
            haversine_km(stops[a].lat, stops[a].lon, stops[b].lat, stops[b].lon)
            for a, b in zip(indices, indices[1:], strict=False)
        ]
        route_idx = len(routes)
        routes.append(route)
        for pos, s_idx in enumerate(route.stops):
            stop_routes[s_idx].append((route_idx, pos))

    for from_id, to_id, seconds in transfers or []:
        a, b = stop_index.get(from_id), stop_index.get(to_id)
        if a is None or b is None or a == b:
            continue
        _upsert_transfer(transfer_lists[a], b, seconds)
        _upsert_transfer(transfer_lists[b], a, seconds)

    return Timetable(
        stops=stops,
        routes=routes,
        stop_routes=stop_routes,
        transfers=transfer_lists,
        stop_index=stop_index,
    )


def _upsert_transfer(entries: list[tuple[int, int]], to_stop: int, seconds: int) -> None:
    for i, (existing_stop, existing_seconds) in enumerate(entries):
        if existing_stop == to_stop:
            if seconds < existing_seconds:
                entries[i] = (to_stop, seconds)
            return
    entries.append((to_stop, seconds))


def _strip_overtaking(trips: list[Trip]) -> list[Trip]:
    """Greedily keep a maximal non-overtaking prefix chain.

    A trip is kept when it arrives no earlier than the last kept trip at every stop.
    Trips are already sorted by first departure.
    """
    if not trips:
        return trips
    kept = [trips[0]]
    for trip in trips[1:]:
        last = kept[-1]
        if all(x <= y for x, y in zip(last.arrivals, trip.arrivals, strict=False)):
            kept.append(trip)
    return kept


def close_transfers(
    transfers: dict[int, dict[int, int]], max_seconds: int
) -> dict[int, dict[int, int]]:
    """Transitive closure of footpaths, capped at `max_seconds`.

    RAPTOR requires footpaths to be transitively closed: it relaxes them exactly once
    per round, so a two-hop walk that is never materialized is a journey the router
    cannot find. Floyd-Warshall over the (small) set of stops that have any footpath.
    """
    nodes = sorted({n for a, targets in transfers.items() for n in (a, *targets)})
    dist: dict[int, dict[int, int]] = {
        a: dict(targets) for a, targets in transfers.items()
    }
    for k in nodes:
        for i in nodes:
            ik = dist.get(i, {}).get(k)
            if ik is None:
                continue
            for j, kj in dist.get(k, {}).items():
                if i == j:
                    continue
                total = ik + kj
                if total > max_seconds:
                    continue
                current = dist.setdefault(i, {}).get(j)
                if current is None or total < current:
                    dist[i][j] = total
    return dist
