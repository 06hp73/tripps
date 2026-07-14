"""Build the routing timetable from a static GTFS feed (GTFS Sweden 3 / GTFS Sverige 2).

Why self-host the feed instead of calling ResRobot's journey planner per query: Trafiklab
rate-limits ResRobot at 45 req/min on the free tier and explicitly declines quota upgrades
for crawler-shaped workloads, pushing bulk consumers to GTFS. A national static feed is
CC0, refreshes daily, and answers an unlimited number of queries offline. ResRobot is kept
only for resolving a typed place name to a stop id.

Two GTFS details drive the code below.

**Extended route types.** Trafiklab feeds use the extended (three- and four-digit) route
type vocabulary, not the basic 0-7 codes. `102` is a long-distance train, `700` a local
bus. Filtering on the basic vocabulary would silently drop every intercity service. Both
are accepted here, because the historical GTFS Sverige 2 feed uses basic codes.

**Regional trains are not optional.** Oresundstag (Goteborg-Malmo) and Malartag
(Stockholm-Orebro) are classified regional/suburban, yet they are frequently the cheapest
way to travel those corridors. A "long-distance only" filter would silently miss the
cheapest answer, so `INTERCITY_ROUTE_TYPES` includes regional rail.
"""

from __future__ import annotations

import csv
import io
import logging
import pickle
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from ..models import Stop, TransportMode
from ..routing.timetable import RouteInfo, Timetable, TimetableBuilder, Trip, close_transfers
from ..timeutil import parse_gtfs_time

log = logging.getLogger(__name__)

#: Bump when the parse output changes shape, so stale pickles are ignored after an upgrade.
_TIMETABLE_CACHE_VERSION = 3

#: Trips from the previous service day that run past midnight are re-expressed relative to this
#: day's midnight; their id is suffixed so the tail is distinguishable from the same train's own
#: same-day run (both can exist in one materialized timetable).
_OVERNIGHT_SUFFIX = "#ovn"

#: One nominal service day in seconds. The whole router treats a service day as a flat 24 h
#: (`from_service_seconds`/`to_service_seconds` do naive-midnight arithmetic and just stamp the
#: zone), so an overnight tail is re-timed by exactly this to stay consistent with every same-day
#: post-midnight trip. A DST-adjusted shift would put overnight trips on a different clock than
#: the rest of the feed - worse, not better.
_DAY_SECONDS = 86400

# --- route type vocabulary -------------------------------------------------
# Extended GTFS route types, per the Google/Trafiklab extension. Ranges rather than
# exhaustive lists, because feeds use codes we have not seen.


def _in(value: int, low: int, high: int) -> bool:
    return low <= value <= high


def mode_for_route_type(route_type: int) -> TransportMode:
    """Map a GTFS route_type onto a planner mode.

    Suburban rail (109), metro (400-404) and local bus (700, 704, 715) are LOCAL_TRANSIT:
    schedules exist for them but no price source does, so they must be distinguishable in
    order to be flagged rather than silently summed into a "cheapest" total.
    """
    # Basic vocabulary (GTFS Sverige 2 and many regional feeds).
    basic = {
        0: TransportMode.LOCAL_TRANSIT,  # tram
        1: TransportMode.LOCAL_TRANSIT,  # subway
        2: TransportMode.TRAIN,
        3: TransportMode.BUS,
        4: TransportMode.FERRY,
        5: TransportMode.LOCAL_TRANSIT,  # cable tram
        6: TransportMode.LOCAL_TRANSIT,  # aerial lift
        7: TransportMode.LOCAL_TRANSIT,  # funicular
    }
    if route_type in basic:
        return basic[route_type]

    if route_type == 109 or _in(route_type, 400, 405):
        return TransportMode.LOCAL_TRANSIT  # suburban rail, metro
    if _in(route_type, 100, 117):
        return TransportMode.TRAIN
    if _in(route_type, 200, 209):
        return TransportMode.BUS  # coach services
    if route_type in (702, 703, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714):
        return TransportMode.BUS  # express / regional coach variants
    if _in(route_type, 700, 716):
        return TransportMode.LOCAL_TRANSIT  # 700 generic, 704 local, 715 demand-responsive
    if _in(route_type, 900, 906):
        return TransportMode.LOCAL_TRANSIT  # tram
    if _in(route_type, 1000, 1021):
        return TransportMode.FERRY
    if route_type == 1100:
        return TransportMode.FLIGHT
    if _in(route_type, 1200, 1299):
        return TransportMode.FERRY
    if _in(route_type, 1500, 1507):
        return TransportMode.LOCAL_TRANSIT  # taxi
    return TransportMode.LOCAL_TRANSIT


#: Route types worth putting in the intercity network. Includes regional and suburban rail
#: because those services carry the cheapest fares on several trunk corridors, and ferries
#: because Destination Gotland is the dominant cheap option to Visby.
INTERCITY_MODES = frozenset(
    {TransportMode.TRAIN, TransportMode.BUS, TransportMode.FERRY, TransportMode.FLIGHT}
)


@dataclass(slots=True)
class GtfsConfig:
    """What to keep from a national feed.

    `include_local_transit` pulls in metro/tram/local bus. They are needed for the first
    and last mile, and they blow the timetable up by an order of magnitude, so they are off
    by default for intercity search.
    """

    include_local_transit: bool = False
    #: Only these agencies, by name. Empty means every agency.
    agencies: frozenset[str] = frozenset()
    #: Cap on GTFS `transfers.txt` walk times; also the closure cap.
    max_transfer_seconds: int = 1800
    #: Fallback change time when the feed does not specify one.
    default_transfer_seconds: int = 300
    #: Collapse platforms onto their parent station. Intercity routing does not care which
    #: platform a train leaves from, and collapsing removes a large class of pointless
    #: intra-station transfers.
    collapse_parent_stations: bool = True

    def wants(self, mode: TransportMode) -> bool:
        if mode in INTERCITY_MODES:
            return True
        return self.include_local_transit


@dataclass(slots=True)
class GtfsStats:
    stops: int = 0
    routes: int = 0
    trips: int = 0
    dropped_trips: int = 0
    agencies: set[str] = field(default_factory=set)
    route_types: dict[int, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    if name not in zf.namelist():
        return []
    with zf.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(text))


def _iter_csv(zf: zipfile.ZipFile, name: str):
    """Stream a member. `stop_times.txt` in a national feed has millions of rows."""
    if name not in zf.namelist():
        return
    with zf.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        yield from csv.DictReader(text)


def _iter_columns(zf: zipfile.ZipFile, name: str, columns: list[str]):
    """Stream a member as positional tuples for a fixed set of columns.

    `csv.DictReader` builds a dict per row; on `stop_times.txt` that is 11 million dicts and
    roughly half the whole feed-load time. Reading with `csv.reader` and pre-resolved column
    indices is about twice as fast and is worth it only on the two giant files.

    Yields `(row_values_for_columns)` tuples, or nothing if a requested column is absent.
    """
    if name not in zf.namelist():
        return
    with zf.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        reader = csv.reader(text)
        try:
            header = next(reader)
        except StopIteration:
            return
        index = {col: i for i, col in enumerate(header)}
        try:
            positions = [index[col] for col in columns]
        except KeyError:
            return
        width = len(header)
        for row in reader:
            if len(row) < width:
                continue
            yield tuple(row[p] for p in positions)


def _parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y%m%d").date()


def active_service_ids(zf: zipfile.ZipFile, service_date: date) -> set[str]:
    """Service ids running on `service_date`, honouring calendar_dates exceptions.

    A feed may express its calendar entirely through `calendar_dates.txt` (no
    `calendar.txt` at all), which is legal GTFS and common for operators with irregular
    service, so neither file may be assumed present.
    """
    weekday = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ][service_date.weekday()]

    active: set[str] = set()
    for row in _read_csv(zf, "calendar.txt"):
        try:
            start, end = _parse_date(row["start_date"]), _parse_date(row["end_date"])
        except (KeyError, ValueError):
            continue
        if start <= service_date <= end and row.get(weekday, "0") == "1":
            active.add(row["service_id"])

    for row in _read_csv(zf, "calendar_dates.txt"):
        try:
            if _parse_date(row["date"]) != service_date:
                continue
        except (KeyError, ValueError):
            continue
        exception = row.get("exception_type", "").strip()
        if exception == "1":
            active.add(row["service_id"])
        elif exception == "2":
            active.discard(row["service_id"])

    return active


def load_timetable(
    zip_path: Path | str,
    service_date: date,
    config: GtfsConfig | None = None,
    *,
    merge_overnight: bool = False,
) -> tuple[Timetable, GtfsStats]:
    """Parse a GTFS zip into a `Timetable` for one service date.

    With `merge_overnight`, trips of the *previous* service day that run past midnight are also
    materialized, truncated to their post-midnight tail and re-timed relative to this day's
    midnight. Without it, an early-morning search misses a night train that departed yesterday
    evening and is still running after 00:00 - it belongs to yesterday's service id, so a naive
    single-day load never sees it.
    """
    config = config or GtfsConfig()
    stats = GtfsStats()
    builder = TimetableBuilder()

    with zipfile.ZipFile(zip_path) as zf:
        agencies = {
            row["agency_id"]: row.get("agency_name", row["agency_id"])
            for row in _read_csv(zf, "agency.txt")
            if row.get("agency_id")
        }
        # A single-agency feed may omit agency_id entirely.
        default_agency = next(iter(agencies.values()), None) if len(agencies) == 1 else None

        kept_routes: dict[str, tuple[RouteInfo, TransportMode]] = {}
        for row in _read_csv(zf, "routes.txt"):
            try:
                route_type = int(row["route_type"])
            except (KeyError, ValueError):
                continue
            mode = mode_for_route_type(route_type)
            if not config.wants(mode):
                continue
            operator = agencies.get(row.get("agency_id", ""), default_agency)
            if config.agencies and (operator or "") not in config.agencies:
                continue
            kept_routes[row["route_id"]] = (
                RouteInfo(id=row["route_id"], mode=mode, operator=operator, route_type=route_type),
                mode,
            )
            stats.route_types[route_type] = stats.route_types.get(route_type, 0) + 1
            if operator:
                stats.agencies.add(operator)

        services = active_service_ids(zf, service_date)
        prev_date = service_date - timedelta(days=1)
        services_prev: set[str] = (
            active_service_ids(zf, prev_date) if merge_overnight else set()
        )

        # (route_id, headsign, runs_today, ran_previous_day). A trip id may run on both days -
        # they are distinct vehicles, and yesterday's run crossing midnight is exactly the tail
        # we want in addition to today's own run.
        kept_trips: dict[str, tuple[str, str | None, bool, bool]] = {}
        for route_id, service_id, trip_id, headsign in _iter_columns(
            zf, "trips.txt", ["route_id", "service_id", "trip_id", "trip_headsign"]
        ):
            if route_id not in kept_routes:
                continue
            today = service_id in services
            prev = service_id in services_prev
            if not (today or prev):
                continue
            kept_trips[trip_id] = (route_id, headsign or None, today, prev)

        stops_by_id, alias = _load_stops(zf, config)

        # Stream stop_times once, gathering only the trips we kept. This is 11M rows and the
        # single most expensive part of loading the feed, so the loop is inlined and hand-
        # tuned: reject on trip_id before touching any other column, since ~99.9% of rows
        # belong to local-transit trips we dropped and building a tuple for them is pure waste.
        # Entries: (sequence, stop_id, arrival, departure, no_board, no_alight) - the last two
        # from pickup_type/drop_off_type, where any nonzero value (1 forbidden, 2/3 arrange
        # ahead) counts as forbidden for a cheapest-trip planner: offering a boarding the
        # operator does not freely sell is a wrong journey, not a bargain.
        trip_times: dict[str, list[tuple[int, str, int, int, bool, bool]]] = defaultdict(list)
        if "stop_times.txt" in zf.namelist():
            with zf.open("stop_times.txt") as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                reader = csv.reader(text)
                header = next(reader, None)
                if header is not None:
                    col = {c: i for i, c in enumerate(header)}
                    ci_trip = col["trip_id"]
                    ci_arr = col["arrival_time"]
                    ci_dep = col["departure_time"]
                    ci_stop = col["stop_id"]
                    ci_seq = col["stop_sequence"]
                    # Optional columns; a feed without them allows everything everywhere.
                    ci_pick = col.get("pickup_type")
                    ci_drop = col.get("drop_off_type")
                    width = len(header)
                    for row in reader:
                        if len(row) < width:
                            continue
                        trip_id = row[ci_trip]
                        if trip_id not in kept_trips:
                            continue
                        stop_id = alias.get(row[ci_stop], row[ci_stop])
                        if stop_id not in stops_by_id:
                            continue
                        try:
                            arrival = parse_gtfs_time(row[ci_arr])
                            departure = parse_gtfs_time(row[ci_dep])
                            sequence = int(row[ci_seq])
                        except (KeyError, ValueError):
                            continue  # a stop_time without times is a non-timepoint; skip it
                        no_board = (
                            ci_pick is not None and row[ci_pick] not in ("", "0")
                        )
                        no_alight = (
                            ci_drop is not None and row[ci_drop] not in ("", "0")
                        )
                        trip_times[trip_id].append(
                            (sequence, stop_id, arrival, departure, no_board, no_alight)
                        )

        # Materialize concrete trips: each kept trip contributes its same-day run and/or, when
        # merging overnight, the post-midnight tail of the previous service day's run. Build the
        # emission list first so `used_stops` reflects exactly what is actually added (a tail
        # that keeps < 2 post-midnight stops contributes no stops).
        emissions: list[tuple[RouteInfo, list[str], Trip]] = []
        for trip_id, entries in trip_times.items():
            if len(entries) < 2:
                stats.dropped_trips += 1
                continue
            entries.sort(key=lambda e: e[0])
            # Collapsing platforms can make consecutive stop_times share a station.
            deduped = _dedupe_consecutive(entries)
            if len(deduped) < 2:
                stats.dropped_trips += 1
                continue
            route_id, headsign, today, prev = kept_trips[trip_id]
            info, _mode = kept_routes[route_id]
            if today:
                nb = tuple(e[4] for e in deduped)
                na = tuple(e[5] for e in deduped)
                emissions.append(
                    (
                        info,
                        [e[1] for e in deduped],
                        Trip(
                            id=trip_id,
                            arrivals=[e[2] for e in deduped],
                            departures=[e[3] for e in deduped],
                            headsign=headsign,
                            # None for the (overwhelmingly common) unrestricted trip keeps
                            # the pickle compact and the router's fast path branch-free.
                            no_board=nb if any(nb) else None,
                            no_alight=na if any(na) else None,
                        ),
                    )
                )
            if prev:
                # Keep every stop still boardable after this day's midnight: departure past
                # midnight suffices (arrival >= departure never holds, so arrival-past-
                # midnight implies it). A stop that ARRIVES before midnight but DEPARTS
                # after - a midnight dwell - is a genuine post-midnight boarding, re-timed
                # with its arrival clamped to 00:00 and marked board-only: its true arrival
                # belongs to yesterday, so offering an alight "today" would be fiction.
                tail = [e for e in deduped if e[3] >= _DAY_SECONDS]
                if len(tail) >= 2:
                    nb = tuple(e[4] for e in tail)
                    na = tuple(
                        e[5] or e[2] < _DAY_SECONDS  # straddle stop: board-only
                        for e in tail
                    )
                    emissions.append(
                        (
                            info,
                            [e[1] for e in tail],
                            Trip(
                                id=f"{trip_id}{_OVERNIGHT_SUFFIX}",
                                arrivals=[max(0, e[2] - _DAY_SECONDS) for e in tail],
                                departures=[e[3] - _DAY_SECONDS for e in tail],
                                headsign=headsign,
                                no_board=nb if any(nb) else None,
                                no_alight=na if any(na) else None,
                            ),
                        )
                    )

        used_stops: set[str] = {sid for _, stop_ids, _ in emissions for sid in stop_ids}
        for stop_id in used_stops:
            builder.add_stop(stops_by_id[stop_id])

        for info, stop_ids, trip in emissions:
            if not _monotonic(trip):
                stats.dropped_trips += 1
                stats.problems.append(f"trip {trip.id}: non-monotonic stop times")
                continue
            builder.add_trip(info, stop_ids, trip)
            stats.trips += 1

        _load_transfers(zf, builder, alias, used_stops, config)

    timetable = builder.build()
    stats.stops = timetable.num_stops
    stats.routes = len(timetable.routes)
    stats.problems.extend(timetable.validate())
    return timetable, stats


def extract_agency_stops(
    zip_path: Path | str, cache_dir: Path | None = None
) -> dict[str, list[str]]:
    """Map every agency to the set of stops it serves, over the FULL feed (all route types).

    The routing timetable is filtered to intercity modes, so a county PTA whose only routes
    are local buses has no stops there. Travel-card coverage needs those stops to define the
    card's region, so this scans the whole feed. Stop ids are collapsed to parent stations,
    exactly as the timetable does, so the ids line up with a routed leg's `from_stop.id`.

    Cached to JSON keyed by the feed's mtime: it is a full pass over `stop_times.txt` (~10 s),
    but the result is a few tens of KB and only changes when the feed does.
    """
    zip_path = Path(zip_path)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        mtime = int(zip_path.stat().st_mtime)
        path = cache_dir / f"agency-stops-v{_TIMETABLE_CACHE_VERSION}-{mtime}.json"
        if path.exists():
            try:
                import json

                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - a corrupt cache must not be fatal
                log.warning("agency-stops cache %s unreadable, recomputing: %s", path, exc)

    served = _compute_agency_stops(zip_path)
    result = {agency: sorted(stops) for agency, stops in served.items()}
    if cache_dir is not None:
        import json

        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write agency-stops cache %s: %s", path, exc)
    return result


def _compute_agency_stops(zip_path: Path) -> dict[str, set[str]]:
    config = GtfsConfig()
    with zipfile.ZipFile(zip_path) as zf:
        agencies = {
            row["agency_id"]: row.get("agency_name", row["agency_id"])
            for row in _read_csv(zf, "agency.txt")
            if row.get("agency_id")
        }
        default_agency = next(iter(agencies.values()), None) if len(agencies) == 1 else None

        # route_id -> agency name, over ALL route types (not just intercity).
        route_agency: dict[str, str] = {}
        for row in _read_csv(zf, "routes.txt"):
            agency = agencies.get(row.get("agency_id", ""), default_agency)
            if row.get("route_id") and agency:
                route_agency[row["route_id"]] = agency

        # trip_id -> agency name.
        trip_agency: dict[str, str] = {}
        for route_id, trip_id in _iter_columns(zf, "trips.txt", ["route_id", "trip_id"]):
            agency = route_agency.get(route_id)
            if agency:
                trip_agency[trip_id] = agency

        stops_by_id, alias = _load_stops(zf, config)

        served: dict[str, set[str]] = defaultdict(set)
        if "stop_times.txt" in zf.namelist():
            with zf.open("stop_times.txt") as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                reader = csv.reader(text)
                header = next(reader, None)
                if header is not None:
                    col = {c: i for i, c in enumerate(header)}
                    ci_trip = col["trip_id"]
                    ci_stop = col["stop_id"]
                    width = len(header)
                    for row in reader:
                        if len(row) < width:
                            continue
                        agency = trip_agency.get(row[ci_trip])
                        if agency is None:
                            continue
                        stop_id = alias.get(row[ci_stop], row[ci_stop])
                        if stop_id in stops_by_id:
                            served[agency].add(stop_id)
    return served


def load_timetable_cached(
    zip_path: Path | str,
    service_date: date,
    config: GtfsConfig | None = None,
    *,
    cache_dir: Path | None = None,
    merge_overnight: bool = False,
) -> Timetable:
    """`load_timetable`, but memoized to disk as a pickle keyed by (feed mtime, date).

    Parsing the 564 MB feed takes ~15 s; the built timetable is ~2 MB and unpickles in well
    under a second. So a date is parsed once and thereafter loaded, which is what makes a
    "cheapest day over the next week" search (one timetable per day) and cold-date searches
    across restarts fast. The feed's mtime is in the key, so a fresh download invalidates the
    cache automatically. `merge_overnight` is part of the key so the two variants never share
    a cache entry.
    """
    zip_path = Path(zip_path)
    if cache_dir is None:
        return load_timetable(zip_path, service_date, config, merge_overnight=merge_overnight)[0]

    cache_dir.mkdir(parents=True, exist_ok=True)
    mtime = int(zip_path.stat().st_mtime)
    ovn = "-ovn" if merge_overnight else ""
    key = f"tt-v{_TIMETABLE_CACHE_VERSION}-{mtime}-{service_date.isoformat()}{ovn}.pkl"
    path = cache_dir / key
    if path.exists():
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - a corrupt cache must not be fatal
            log.warning("timetable cache %s unreadable, reparsing: %s", path, exc)

    timetable, _stats = load_timetable(
        zip_path, service_date, config, merge_overnight=merge_overnight
    )
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("wb") as fh:
            pickle.dump(timetable, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic, so a crashed write never leaves a half file
    except Exception as exc:  # noqa: BLE001 - failing to cache is not failing to load
        log.warning("could not write timetable cache %s: %s", path, exc)
    return timetable


def _dedupe_consecutive(
    entries: list[tuple[int, str, int, int, bool, bool]],
) -> list[tuple[int, str, int, int, bool, bool]]:
    """Collapse repeated stations caused by parent-station aliasing.

    Keeps the first arrival and the last departure of the run, which is the correct
    behaviour for a train that calls at two platforms of the same station. The boarding
    flags follow the events they describe: alighting happens at the kept (first) arrival,
    so its no_alight stays; boarding happens at the kept (last) departure, so that row's
    no_board wins.
    """
    out: list[tuple[int, str, int, int, bool, bool]] = []
    for entry in entries:
        if out and out[-1][1] == entry[1]:
            seq, stop_id, arrival, _dep, _no_board, no_alight = out[-1]
            out[-1] = (seq, stop_id, arrival, entry[3], entry[4], no_alight)
        else:
            out.append(entry)
    return out


def _monotonic(trip: Trip) -> bool:
    for i in range(len(trip.arrivals)):
        if trip.departures[i] < trip.arrivals[i]:
            return False
        if i + 1 < len(trip.arrivals) and trip.arrivals[i + 1] < trip.departures[i]:
            return False
    return True


def _load_stops(
    zf: zipfile.ZipFile, config: GtfsConfig
) -> tuple[dict[str, Stop], dict[str, str]]:
    """Return (stops by effective id, alias from raw stop_id to effective id)."""
    raw_rows = _read_csv(zf, "stops.txt")
    alias: dict[str, str] = {}
    stops: dict[str, Stop] = {}

    parents = {
        row["stop_id"]
        for row in raw_rows
        if row.get("location_type", "0") == "1"
    }

    for row in raw_rows:
        stop_id = row["stop_id"]
        location_type = row.get("location_type", "0") or "0"
        if location_type not in ("0", "1"):
            continue  # entrances, generic nodes, boarding areas are not routable
        parent = (row.get("parent_station") or "").strip()
        if config.collapse_parent_stations and parent and parent in parents:
            alias[stop_id] = parent
            continue
        try:
            lat, lon = float(row["stop_lat"]), float(row["stop_lon"])
        except (KeyError, ValueError):
            continue
        stops[stop_id] = Stop(
            id=stop_id,
            name=row.get("stop_name") or stop_id,
            lat=lat,
            lon=lon,
            external_ids={"gtfs": stop_id},
        )
    return stops, alias


def _load_transfers(
    zf: zipfile.ZipFile,
    builder: TimetableBuilder,
    alias: dict[str, str],
    used_stops: set[str],
    config: GtfsConfig,
) -> None:
    """Load `transfers.txt`, symmetrize it, then transitively close the footpath graph.

    RAPTOR relaxes footpaths exactly once per round, so a two-hop walk that is never
    materialized is a journey the router simply cannot find. Closure is capped so the
    walk graph does not turn into an all-pairs matrix.

    `transfers.txt` rows are directional, and feeds routinely list only one direction of a
    pair. A footpath is physically symmetric: if you can walk from the train station to the
    coach terminal in ten minutes, you can walk back. Loading it directionally strands the
    return journey, so pairs are mirrored here. Rows that forbid a transfer
    (`transfer_type=3`) are dropped before mirroring, which keeps that veto effective.
    """
    raw: dict[int, dict[int, int]] = defaultdict(dict)
    index: dict[str, int] = {}

    def idx(stop_id: str) -> int | None:
        if stop_id not in used_stops:
            return None
        if stop_id not in index:
            index[stop_id] = builder.index_of(stop_id)
        return index[stop_id]

    for row in _iter_csv(zf, "transfers.txt"):
        from_id = alias.get(row.get("from_stop_id", ""), row.get("from_stop_id", ""))
        to_id = alias.get(row.get("to_stop_id", ""), row.get("to_stop_id", ""))
        if from_id == to_id:
            continue
        a, b = idx(from_id), idx(to_id)
        if a is None or b is None:
            continue
        transfer_type = (row.get("transfer_type") or "0").strip()
        if transfer_type == "3":
            continue  # transfer explicitly forbidden here
        try:
            seconds = int(row.get("min_transfer_time") or config.default_transfer_seconds)
        except ValueError:
            seconds = config.default_transfer_seconds
        if seconds > config.max_transfer_seconds:
            continue
        for x, y in ((a, b), (b, a)):
            current = raw[x].get(y)
            if current is None or seconds < current:
                raw[x][y] = seconds

    closed = close_transfers(raw, config.max_transfer_seconds)
    for a, targets in closed.items():
        for b, seconds in targets.items():
            builder.add_transfer(
                builder.stop_at(a).id, builder.stop_at(b).id, seconds, bidirectional=False
            )
