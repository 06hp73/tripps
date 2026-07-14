"""Conversion between wall-clock datetimes and RAPTOR's service-day seconds.

GTFS expresses stop times as an offset from *local midnight* of the service day, and
allows the offset to exceed 24 hours so that a 23:50 departure arriving at 00:30 reads
as `24:30:00` on the same service day. The routing core therefore works in integer
"seconds since local midnight", not in absolute epoch seconds.

The consequence is that arithmetic here is deliberately naive-local. On the two DST
transition days a service day is 23 or 25 hours long, and an offset of 86400 is not
exactly 24 hours of elapsed time. GTFS itself has this property, so matching it keeps us
consistent with the feed; the alternative (absolute seconds) would silently shift every
timetabled departure by an hour twice a year.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .models import STOCKHOLM_TZ

TZ = ZoneInfo(STOCKHOLM_TZ)


def parse_gtfs_time(value: str) -> int:
    """`"25:10:00"` -> 90600. Rejects anything that is not H:MM:SS."""
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"bad GTFS time: {value!r}")
    try:
        hours, minutes, seconds = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"bad GTFS time: {value!r}") from exc
    if minutes >= 60 or seconds >= 60 or hours < 0 or minutes < 0 or seconds < 0:
        raise ValueError(f"bad GTFS time: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def format_service_seconds(seconds: int) -> str:
    """Inverse of `parse_gtfs_time`, keeping hours past 24 visible."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def to_service_seconds(moment: datetime, service_date: date) -> int:
    """Wall-clock offset of `moment` from local midnight of `service_date`.

    `moment` may be aware or naive; an aware value is first expressed in Europe/Stockholm.
    A moment on the following calendar day yields a value above 86400, which is exactly
    what the timetable expects.
    """
    if moment.tzinfo is not None:
        moment = moment.astimezone(TZ)
    naive = moment.replace(tzinfo=None)
    midnight = datetime.combine(service_date, time.min)
    return int((naive - midnight).total_seconds())


def from_service_seconds(seconds: int, service_date: date) -> datetime:
    """Inverse of `to_service_seconds`, returning an aware Europe/Stockholm datetime.

    KNOWN, ACCEPTED EDGE (spring-forward night only): a nominal offset landing in the
    skipped 02:00-03:00 hour yields a wall label that does not exist, carrying the pre-gap
    offset (PEP 495 fold=0). All internal comparison, validation, and duration arithmetic is
    same-tzinfo and therefore naive-wall (correct against the GTFS nominal timeline), but a
    serialized ISO string from inside the gap denotes a UTC instant that can invert against a
    neighbouring leg's - an external client must not do cross-leg instant math on that one
    hour a year. Normalizing the label here instead would break the naive-wall harmony every
    internal check relies on (a 03:50 -> 03:10 gap-straddling leg would suddenly fail the
    arrival >= departure wall check), trading a documented cosmetic artifact for real
    crashes; see the module docstring for why nominal-local is the design.
    """
    midnight = datetime.combine(service_date, time.min)
    return (midnight + timedelta(seconds=int(seconds))).replace(tzinfo=TZ)


def now_local() -> datetime:
    return datetime.now(TZ)


def service_dates_covering(service_date: date) -> list[date]:
    """Service days whose stop times can appear on `service_date`.

    A trip on the previous service day running past midnight (`24:xx:xx`) is still in
    service on this calendar date, so both days must be scanned when materializing a
    timetable for a query.
    """
    return [service_date - timedelta(days=1), service_date]
