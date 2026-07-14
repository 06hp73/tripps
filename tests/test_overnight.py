"""Overnight service-day merge: a night train that departed yesterday and is still running
after midnight must be visible to an early-morning search on the arrival date.

The trap the merge avoids: that train belongs to yesterday's service id, so a naive single-day
load for the arrival date never sees it, silently hiding a real journey.
"""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

from tripps.ingest.gtfs import load_timetable

# Three stations on a southern night line. NIGHT departs Malmö at 22:00 and reaches Norrköping
# 00:30 and Stockholm 02:00 the next calendar day. DAY is an ordinary daytime train, present to
# prove same-day trips are neither dropped nor duplicated by the merge.
_STOPS = """stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
MMX,Malmo C,55.6090,13.0000,0,
NRK,Norrkoping C,58.5960,16.1830,0,
STO,Stockholm C,59.3300,18.0590,0,
"""

_ROUTES = """route_id,agency_id,route_short_name,route_long_name,route_type
R_NIGHT,SJ,,Malmo-Stockholm night,102
R_DAY,SJ,,Malmo-Stockholm day,102
"""

_AGENCY = """agency_id,agency_name,agency_url,agency_timezone
SJ,SJ,https://sj.se,Europe/Stockholm
"""

# DAY runs daily; NIGHT is switched on for a single date via calendar_dates, so its presence on
# the arrival date can only come from the overnight merge.
_CALENDAR = """service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DAILY,1,1,1,1,1,1,1,20260101,20261231
"""


def _feed(tmp_path: Path, night_dates: list[date]) -> Path:
    cal_dates = "service_id,date,exception_type\n" + "".join(
        f"NIGHTSVC,{d.strftime('%Y%m%d')},1\n" for d in night_dates
    )
    # NIGHT: MMX 22:00, NRK 24:30/24:35, STO 26:00 (00:30/00:35/02:00 next day).
    stop_times = (
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T_NIGHT,22:00:00,22:00:00,MMX,1\n"
        "T_NIGHT,24:30:00,24:35:00,NRK,2\n"
        "T_NIGHT,26:00:00,26:00:00,STO,3\n"
        "T_DAY,08:00:00,08:00:00,MMX,1\n"
        "T_DAY,10:00:00,10:00:00,NRK,2\n"
        "T_DAY,12:00:00,12:00:00,STO,3\n"
    )
    trips = (
        "route_id,service_id,trip_id,trip_headsign\n"
        "R_NIGHT,NIGHTSVC,T_NIGHT,Stockholm\n"
        "R_DAY,DAILY,T_DAY,Stockholm\n"
    )
    path = tmp_path / "night.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("agency.txt", _AGENCY)
        zf.writestr("stops.txt", _STOPS)
        zf.writestr("routes.txt", _ROUTES)
        zf.writestr("calendar.txt", _CALENDAR)
        zf.writestr("calendar_dates.txt", cal_dates)
        zf.writestr("trips.txt", trips)
        zf.writestr("stop_times.txt", stop_times)
    return path


def _trips_by_id(tt) -> dict[str, tuple[list[str], list[int], list[int]]]:
    """trip id -> (stop id path, departures, arrivals) across every route."""
    out = {}
    for route in tt.routes:
        path = [tt.stops[i].id for i in route.stops]
        for trip in route.trips:
            out[trip.id] = (path, list(trip.departures), list(trip.arrivals))
    return out


_DAY_SECONDS = 86400  # the nominal service day the router re-times an overnight tail by


NIGHT_DAY = date(2026, 7, 15)  # a Wednesday
NEXT_DAY = date(2026, 7, 16)


def test_night_train_absent_without_merge(tmp_path):
    feed = _feed(tmp_path, [NIGHT_DAY])
    tt, _ = load_timetable(feed, NEXT_DAY, merge_overnight=False)
    ids = _trips_by_id(tt)
    assert "T_DAY" in ids  # the ordinary daytime train is there
    assert not any(k.startswith("T_NIGHT") for k in ids), "night train leaked without merge"


def test_night_train_tail_present_with_merge(tmp_path):
    feed = _feed(tmp_path, [NIGHT_DAY])
    tt, _ = load_timetable(feed, NEXT_DAY, merge_overnight=True)
    ids = _trips_by_id(tt)

    assert "T_NIGHT#ovn" in ids, "post-midnight tail of yesterday's night train is missing"
    path, deps, arrs = ids["T_NIGHT#ovn"]
    # The pre-midnight Malmö stop is dropped; only the post-midnight tail remains.
    assert path == ["NRK", "STO"]
    # Norrköping departure 24:35 -> 00:35 relative to the arrival date's midnight.
    assert deps[0] == 24 * 3600 + 35 * 60 - _DAY_SECONDS
    assert arrs[0] == 24 * 3600 + 30 * 60 - _DAY_SECONDS
    assert arrs[1] == 26 * 3600 - _DAY_SECONDS  # Stockholm 02:00
    # The ordinary train is still present exactly once.
    assert "T_DAY" in ids


def test_same_day_run_unaffected_and_not_duplicated(tmp_path):
    feed = _feed(tmp_path, [NIGHT_DAY])
    tt, _ = load_timetable(feed, NIGHT_DAY, merge_overnight=True)
    ids = _trips_by_id(tt)
    # On its own service date the night train appears in full, boarding at Malmö...
    assert "T_NIGHT" in ids
    assert ids["T_NIGHT"][0] == ["MMX", "NRK", "STO"]
    # ...and there is no spurious tail (the previous day had no NIGHT service).
    assert "T_NIGHT#ovn" not in ids


def test_daily_train_not_duplicated_by_merge(tmp_path):
    """A daytime train runs on both days but has no post-midnight stops, so the merge must add
    no tail for it - it appears exactly once."""
    feed = _feed(tmp_path, [NIGHT_DAY])
    tt, _ = load_timetable(feed, NEXT_DAY, merge_overnight=True)
    day_trips = [t for t in _trips_by_id(tt) if t.startswith("T_DAY")]
    assert day_trips == ["T_DAY"]


def test_overnight_tail_uses_nominal_day_across_dst(tmp_path):
    """The re-timing is a flat nominal service day (86400 s) even across a DST-change night.

    This is deliberate and correct: `from_service_seconds`/`to_service_seconds`, which the whole
    router and the UI use to turn service seconds into wall-clock, do naive-midnight arithmetic
    and merely stamp the zone - their round-trip is always S-86400. Re-timing an overnight tail
    by a DST-adjusted amount would put it on a different clock than every same-day post-midnight
    trip, so displayed times would drift. Consistency beats a clever adjustment nothing else
    honours."""
    arrival = date(2026, 3, 30)  # the morning after EU clocks jump forward
    night_before = date(2026, 3, 29)
    feed = _feed(tmp_path, [night_before])
    tt, _ = load_timetable(feed, arrival, merge_overnight=True)
    ids = _trips_by_id(tt)
    assert "T_NIGHT#ovn" in ids
    _, deps, _ = ids["T_NIGHT#ovn"]
    assert deps[0] == 24 * 3600 + 35 * 60 - _DAY_SECONDS


def test_midnight_dwell_stop_is_board_only_in_the_tail(tmp_path):
    """A stop the night train REACHES before midnight but LEAVES after (arr 23:50, dep 00:10)
    is a genuine post-midnight boarding on the arrival date. The old AND-filter dropped it
    from the tail entirely. It must appear board-only: its true arrival belongs to yesterday,
    so alighting 'today' there would be fiction."""
    stop_times = (
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T_NIGHT,22:00:00,22:00:00,MMX,1\n"
        "T_NIGHT,23:50:00,24:10:00,NRK,2\n"  # the midnight dwell
        "T_NIGHT,26:00:00,26:00:00,STO,3\n"
        "T_DAY,08:00:00,08:00:00,MMX,1\n"
        "T_DAY,10:00:00,10:00:00,NRK,2\n"
        "T_DAY,12:00:00,12:00:00,STO,3\n"
    )
    trips = (
        "route_id,service_id,trip_id,trip_headsign\n"
        "R_NIGHT,NIGHTSVC,T_NIGHT,Stockholm\n"
        "R_DAY,DAILY,T_DAY,Stockholm\n"
    )
    cal_dates = f"service_id,date,exception_type\nNIGHTSVC,{NIGHT_DAY.strftime('%Y%m%d')},1\n"
    path = tmp_path / "dwell.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("agency.txt", _AGENCY)
        zf.writestr("stops.txt", _STOPS)
        zf.writestr("routes.txt", _ROUTES)
        zf.writestr("calendar.txt", _CALENDAR)
        zf.writestr("calendar_dates.txt", cal_dates)
        zf.writestr("trips.txt", trips)
        zf.writestr("stop_times.txt", stop_times)

    tt, _ = load_timetable(path, NEXT_DAY, merge_overnight=True)
    tail = None
    for route in tt.routes:
        for trip in route.trips:
            if trip.id == "T_NIGHT#ovn":
                tail = (route, trip)
    assert tail is not None, "the dwell tail must exist at all"
    route, trip = tail
    assert [tt.stops[i].id for i in route.stops] == ["NRK", "STO"]
    assert trip.departures[0] == 10 * 60  # boards NRK at 00:10 on the arrival date
    assert trip.arrivals[0] == 0  # clamped: the true arrival belongs to yesterday
    assert trip.no_alight == (True, False), "dwell stop is board-only"
    assert trip.no_board is None
