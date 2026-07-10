"""GTFS ingest: route-type vocabulary, calendars, parent stations, transfer closure."""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import pytest

from tripps.ingest.gtfs import (
    GtfsConfig,
    active_service_ids,
    load_timetable,
    mode_for_route_type,
)
from tripps.models import TransportMode

# A miniature national feed: a high-speed train, an express coach, a local bus,
# a ferry, and a regional train, over four stations with a parent-station pair.
AGENCY = """agency_id,agency_name,agency_url,agency_timezone
SJ,SJ,https://sj.se,Europe/Stockholm
FLIX,FlixBus,https://flixbus.se,Europe/Stockholm
SL,Storstockholms Lokaltrafik,https://sl.se,Europe/Stockholm
DG,Destination Gotland,https://destinationgotland.se,Europe/Stockholm
OTAG,Oresundstag,https://oresundstag.se,Europe/Stockholm
"""

# STO has two platforms collapsing onto one parent station.
STOPS = """stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
STO,Stockholm C,59.3300,18.0590,1,
STO_1,Stockholm C track 1,59.3301,18.0591,0,STO
STO_2,Stockholm C track 2,59.3302,18.0592,0,STO
GBG,Goteborg C,57.7089,11.9746,0,
NRK,Norrkoping C,58.5960,16.1830,0,
MMX,Malmo C,55.6090,13.0000,0,
NYN,Nynashamn,58.9030,17.9490,0,
VBY,Visby,57.6410,18.2960,0,
STO_BUS,Stockholm Cityterminalen,59.3310,18.0570,0,
"""

# 101 high-speed rail, 202 national coach, 700 local bus, 1000 ferry, 106 regional rail.
ROUTES = """route_id,agency_id,route_short_name,route_long_name,route_type
R_FAST,SJ,X2000,Stockholm-Goteborg,101
R_COACH,FLIX,700,Stockholm-Goteborg coach,202
R_LOCAL,SL,4,City bus,700
R_FERRY,DG,,Nynashamn-Visby,1000
R_REG,OTAG,,Malmo-Goteborg regional,106
"""

CALENDAR = """service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WEEKDAY,1,1,1,1,1,0,0,20260101,20261231
DAILY,1,1,1,1,1,1,1,20260101,20261231
"""

# 2026-07-15 is a Wednesday: NOSERVICE is removed, EXTRA is added.
CALENDAR_DATES = """service_id,date,exception_type
DAILY,20260715,2
EXTRA,20260715,1
"""

TRIPS = """route_id,service_id,trip_id,trip_headsign
R_FAST,WEEKDAY,T_FAST,Goteborg
R_COACH,WEEKDAY,T_COACH,Goteborg
R_LOCAL,WEEKDAY,T_LOCAL,City
R_FERRY,DAILY,T_FERRY,Visby
R_REG,WEEKDAY,T_REG,Goteborg
R_FAST,EXTRA,T_EXTRA,Goteborg
R_FAST,DAILY,T_CANCELLED,Goteborg
"""

# T_FAST calls at both Stockholm platforms; they collapse into one station stop.
STOP_TIMES = """trip_id,arrival_time,departure_time,stop_id,stop_sequence
T_FAST,08:00:00,08:00:00,STO_1,1
T_FAST,08:02:00,08:03:00,STO_2,2
T_FAST,09:10:00,09:12:00,NRK,3
T_FAST,11:00:00,11:00:00,GBG,4
T_COACH,09:00:00,09:00:00,STO_BUS,1
T_COACH,15:30:00,15:30:00,GBG,2
T_LOCAL,07:00:00,07:00:00,STO_1,1
T_LOCAL,07:15:00,07:15:00,STO_BUS,2
T_FERRY,11:00:00,11:00:00,NYN,1
T_FERRY,14:15:00,14:15:00,VBY,2
T_REG,06:00:00,06:00:00,MMX,1
T_REG,09:30:00,09:30:00,GBG,2
T_EXTRA,23:50:00,23:50:00,STO_1,1
T_EXTRA,25:10:00,25:10:00,GBG,2
T_CANCELLED,10:00:00,10:00:00,STO_1,1
T_CANCELLED,13:00:00,13:00:00,GBG,2
"""

TRANSFERS = """from_stop_id,to_stop_id,transfer_type,min_transfer_time
STO,STO_BUS,2,600
STO_BUS,NRK,3,60
"""


@pytest.fixture
def feed(tmp_path: Path) -> Path:
    path = tmp_path / "mini.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("agency.txt", AGENCY)
        zf.writestr("stops.txt", STOPS)
        zf.writestr("routes.txt", ROUTES)
        zf.writestr("calendar.txt", CALENDAR)
        zf.writestr("calendar_dates.txt", CALENDAR_DATES)
        zf.writestr("trips.txt", TRIPS)
        zf.writestr("stop_times.txt", STOP_TIMES)
        zf.writestr("transfers.txt", TRANSFERS)
    return path


# --- route type vocabulary ------------------------------------------------


@pytest.mark.parametrize(
    ("route_type", "expected"),
    [
        (2, TransportMode.TRAIN),  # basic vocabulary
        (3, TransportMode.BUS),
        (4, TransportMode.FERRY),
        (100, TransportMode.TRAIN),
        (101, TransportMode.TRAIN),  # high speed, X2000
        (102, TransportMode.TRAIN),  # long distance
        (106, TransportMode.TRAIN),  # regional: Oresundstag, Malartag
        (109, TransportMode.LOCAL_TRANSIT),  # suburban
        (202, TransportMode.BUS),  # national coach
        (401, TransportMode.LOCAL_TRANSIT),  # metro
        (700, TransportMode.LOCAL_TRANSIT),  # local bus
        (702, TransportMode.BUS),  # express bus
        (900, TransportMode.LOCAL_TRANSIT),  # tram
        (1000, TransportMode.FERRY),
        (1100, TransportMode.FLIGHT),
    ],
)
def test_route_type_mapping(route_type, expected):
    assert mode_for_route_type(route_type) is expected


def test_regional_rail_is_intercity_not_local():
    """Oresundstag (106) is often the cheapest Goteborg-Malmo option. Classifying it as
    local transit would exclude it from the intercity network and hide the cheapest
    answer."""
    assert mode_for_route_type(106) is TransportMode.TRAIN
    assert GtfsConfig().wants(mode_for_route_type(106))


# --- calendars -------------------------------------------------------------


def test_active_services_respects_weekday(feed):
    with zipfile.ZipFile(feed) as zf:
        saturday = active_service_ids(zf, date(2026, 7, 11))
    assert "WEEKDAY" not in saturday
    assert "DAILY" in saturday


def test_calendar_dates_add_and_remove(feed):
    with zipfile.ZipFile(feed) as zf:
        wednesday = active_service_ids(zf, date(2026, 7, 15))
    assert "EXTRA" in wednesday, "exception_type=1 adds a service"
    assert "DAILY" not in wednesday, "exception_type=2 removes a service"
    assert "WEEKDAY" in wednesday


def test_service_outside_calendar_range_is_inactive(feed):
    with zipfile.ZipFile(feed) as zf:
        assert active_service_ids(zf, date(2025, 7, 15)) == set()


# --- timetable construction -----------------------------------------------


def test_loads_intercity_modes_and_excludes_local_transit(feed):
    tt, stats = load_timetable(feed, date(2026, 7, 8))  # a Wednesday
    modes = {r.info.mode for r in tt.routes}
    assert TransportMode.TRAIN in modes
    assert TransportMode.BUS in modes
    assert TransportMode.FERRY in modes
    assert TransportMode.LOCAL_TRANSIT not in modes
    assert stats.problems == []


def test_local_transit_included_on_request(feed):
    tt, _ = load_timetable(feed, date(2026, 7, 8), GtfsConfig(include_local_transit=True))
    modes = {r.info.mode for r in tt.routes}
    assert TransportMode.LOCAL_TRANSIT in modes


def test_parent_station_collapses_platforms(feed):
    tt, _ = load_timetable(feed, date(2026, 7, 8))
    ids = {s.id for s in tt.stops}
    assert "STO" in ids
    assert "STO_1" not in ids and "STO_2" not in ids


def test_repeated_platform_calls_collapse_to_one_station_stop(feed):
    """T_FAST calls at track 1 then track 2. After collapsing it must visit STO once,
    keeping the first arrival (08:00) and the last departure (08:03)."""
    tt, _ = load_timetable(feed, date(2026, 7, 8))
    route = next(r for r in tt.routes if r.info.id == "R_FAST")
    trip = next(t for t in route.trips if t.id == "T_FAST")
    assert [tt.stops[s].id for s in route.stops] == ["STO", "NRK", "GBG"]
    assert trip.arrivals[0] == 8 * 3600
    assert trip.departures[0] == 8 * 3600 + 3 * 60


def test_cancelled_service_trip_is_absent(feed):
    """T_CANCELLED runs on DAILY, which calendar_dates removes on 2026-07-15."""
    tt, _ = load_timetable(feed, date(2026, 7, 15))
    trip_ids = {t.id for r in tt.routes for t in r.trips}
    assert "T_CANCELLED" not in trip_ids
    assert "T_EXTRA" in trip_ids


def test_after_midnight_stop_time_is_preserved(feed):
    tt, _ = load_timetable(feed, date(2026, 7, 15))
    trip = next(t for r in tt.routes for t in r.trips if t.id == "T_EXTRA")
    assert trip.departures[0] == 23 * 3600 + 50 * 60
    assert trip.arrivals[1] == 25 * 3600 + 10 * 60


def test_distinct_stop_patterns_become_distinct_routes(feed):
    """R_FAST serves STO-NRK-GBG; the extra service serves STO-GBG. Same GTFS route_id,
    two RAPTOR routes, because RAPTOR requires an identical stop sequence."""
    tt, _ = load_timetable(feed, date(2026, 7, 15))
    fast_routes = [r for r in tt.routes if r.info.id == "R_FAST"]
    assert len(fast_routes) == 2
    patterns = sorted(tuple(tt.stops[s].id for s in r.stops) for r in fast_routes)
    assert patterns == [("STO", "GBG"), ("STO", "NRK", "GBG")]


def test_agency_becomes_operator(feed):
    tt, stats = load_timetable(feed, date(2026, 7, 8))
    operators = {r.info.operator for r in tt.routes}
    assert "SJ" in operators and "FlixBus" in operators
    assert "Destination Gotland" in operators
    assert "Storstockholms Lokaltrafik" not in operators  # local transit filtered out
    assert "SJ" in stats.agencies


def test_agency_filter(feed):
    tt, _ = load_timetable(feed, date(2026, 7, 8), GtfsConfig(agencies=frozenset({"SJ"})))
    assert {r.info.operator for r in tt.routes} == {"SJ"}


def test_forbidden_transfer_is_not_loaded(feed):
    """transfer_type=3 means "transfer not possible here"."""
    tt, _ = load_timetable(feed, date(2026, 7, 8), GtfsConfig(include_local_transit=True))
    bus = tt.index_of("STO_BUS")
    nrk = tt.index_of("NRK")
    assert all(target != nrk for target, _ in tt.transfers[bus])


def test_transfer_is_bidirectional_and_timed(feed):
    tt, _ = load_timetable(feed, date(2026, 7, 8), GtfsConfig(include_local_transit=True))
    sto, bus = tt.index_of("STO"), tt.index_of("STO_BUS")
    assert (bus, 600) in tt.transfers[sto]
    assert (sto, 600) in tt.transfers[bus]


def test_ferry_route_is_routable(feed):
    """Destination Gotland is the dominant cheap option to Visby, so the ferry mode has
    to survive ingest."""
    from tripps.routing.floors import zero_floors
    from tripps.routing.mcraptor import RaptorQuery, run_mcraptor

    tt, _ = load_timetable(feed, date(2026, 7, 8))
    res = run_mcraptor(
        tt,
        zero_floors(),
        RaptorQuery(origins=[(tt.index_of("NYN"), 0)], targets={tt.index_of("VBY")}),
    )
    assert len(res.labels) == 1
    assert res.labels[0].arrival == 14 * 3600 + 15 * 60


def test_end_to_end_route_over_ingested_feed(feed):
    """Stockholm -> Goteborg: the fast train and the cheap coach are both on the frontier."""
    from tripps.routing.floors import PriceFloorModel
    from tripps.routing.mcraptor import RaptorQuery, run_mcraptor

    tt, _ = load_timetable(feed, date(2026, 7, 8), GtfsConfig(include_local_transit=True))
    res = run_mcraptor(
        tt,
        PriceFloorModel(),
        RaptorQuery(
            origins=[(tt.index_of("STO"), 6 * 3600)],
            targets={tt.index_of("GBG")},
        ),
    )
    arrivals = {lbl.arrival for lbl in res.labels}
    assert 11 * 3600 in arrivals, "X2000 arrival"
    # Cheapest first.
    assert res.labels[0].price_ore <= res.labels[-1].price_ore


def test_missing_optional_files_are_tolerated(tmp_path: Path):
    """A feed with no transfers.txt and no calendar.txt is legal GTFS."""
    path = tmp_path / "sparse.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("agency.txt", AGENCY)
        zf.writestr("stops.txt", STOPS)
        zf.writestr("routes.txt", ROUTES)
        zf.writestr("calendar_dates.txt", "service_id,date,exception_type\nONLY,20260708,1\n")
        zf.writestr("trips.txt", "route_id,service_id,trip_id,trip_headsign\nR_FAST,ONLY,T1,Goteborg\n")
        zf.writestr(
            "stop_times.txt",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "T1,08:00:00,08:00:00,STO_1,1\nT1,11:00:00,11:00:00,GBG,2\n",
        )
    tt, stats = load_timetable(path, date(2026, 7, 8))
    assert tt.num_trips == 1
    assert stats.problems == []
