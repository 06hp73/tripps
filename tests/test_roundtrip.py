"""Round-trip pairing, cheapest-day-over-window, and the disk-cached timetable loader."""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import pytest

from tripps.config import CacheTTL, PricingBudget
from tripps.db import Database
from tripps.ingest.gtfs import load_timetable, load_timetable_cached
from tripps.models import ADULT, PriceConfidence, Quote, Stop, TransportMode  # noqa: E402
from tripps.pricing.orchestrator import PricingOrchestrator
from tripps.routing.floors import PriceFloorModel
from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip
from tripps.search import Planner, SearchOptions, cheapest_over_window, round_trip

from .test_gtfs import (  # reuse the mini-feed builder
    AGENCY,
    CALENDAR,
    CALENDAR_DATES,
    ROUTES,
    STOP_TIMES,
    STOPS,
    TRANSFERS,
    TRIPS,
)


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "rt.sqlite3")
    yield database
    database.close()


class PriceByOperator:
    name = "stub"
    modes = frozenset({TransportMode.TRAIN, TransportMode.BUS})
    provides_price = True

    def __init__(self, fares):
        self.fares = fares

    def supports(self, leg):
        return (leg.operator or "") in self.fares

    async def quote_leg(self, leg, passenger=ADULT):
        return Quote(source="stub", amount_ore=self.fares[leg.operator], confidence=PriceConfidence.EXACT)

    async def health(self):
        from tripps.interfaces import HealthState, SourceHealth

        return SourceHealth(self.name, HealthState.OK)

    async def aclose(self):
        pass


def _stop(sid, name, lat, lon):
    return Stop(id=sid, name=name, lat=lat, lon=lon)


def _tt(routes):
    b = TimetableBuilder()
    for rid, mode, op, stops, times in routes:
        for s in stops:
            b.add_stop(s)
        b.add_trip(
            RouteInfo(id=rid, mode=mode, operator=op),
            [s.id for s in stops],
            Trip(id=f"{rid}-t", arrivals=[t for t, _ in times], departures=[d for _, d in times]),
        )
    return b.build()


def _hhmm(h, m=0):
    return h * 3600 + m * 60


def _planner_for(db, timetable, fares):
    orch = PricingOrchestrator(
        [PriceByOperator(fares)], db,
        budget=PricingBudget(min_interval_seconds=0.0), ttl=CacheTTL(), floors=PriceFloorModel(),
    )
    return Planner(timetable, orch, db=db, options=SearchOptions(include_flights=False))


# --- round trip ------------------------------------------------------------


async def test_round_trip_searches_both_directions_and_sums(db):
    sto = _stop("STO", "Stockholm", 59.33, 18.06)
    gbg = _stop("GBG", "Goteborg", 57.71, 11.97)
    out_tt = _tt([("O", TransportMode.TRAIN, "SJ", [sto, gbg], [(_hhmm(8),) * 2, (_hhmm(11),) * 2])])
    in_tt = _tt([("R", TransportMode.TRAIN, "SJ", [gbg, sto], [(_hhmm(14),) * 2, (_hhmm(17),) * 2])])
    planners = {date(2026, 7, 20): out_tt, date(2026, 7, 27): in_tt}

    def make_planner(d):
        return _planner_for(db, planners[d], {"SJ": 30_000})

    result = await round_trip(
        make_planner, "Stockholm", "Goteborg", date(2026, 7, 20), date(2026, 7, 27)
    )
    assert result.cheapest_outbound.total_price_ore == 30_000
    assert result.cheapest_inbound.total_price_ore == 30_000
    assert result.total_price_ore == 60_000


async def test_round_trip_total_is_none_if_a_direction_is_unpriced(db):
    sto = _stop("STO", "Stockholm", 59.33, 18.06)
    gbg = _stop("GBG", "Goteborg", 57.71, 11.97)
    out_tt = _tt([("O", TransportMode.TRAIN, "SJ", [sto, gbg], [(_hhmm(8),) * 2, (_hhmm(11),) * 2])])
    in_tt = _tt([("R", TransportMode.TRAIN, "Ybuss", [gbg, sto], [(_hhmm(14),) * 2, (_hhmm(17),) * 2])])
    planners = {date(2026, 7, 20): out_tt, date(2026, 7, 27): in_tt}

    def make_planner(d):
        # Only SJ is priceable; the return leg on Ybuss stays unpriced.
        return _planner_for(db, planners[d], {"SJ": 30_000})

    result = await round_trip(
        make_planner, "Stockholm", "Goteborg", date(2026, 7, 20), date(2026, 7, 27)
    )
    assert result.cheapest_outbound is not None
    assert result.cheapest_inbound is None  # nothing fully priced that way
    assert result.total_price_ore is None


# --- cheapest over window --------------------------------------------------


async def test_cheapest_over_window_returns_one_fare_per_day(db):
    sto = _stop("STO", "Stockholm", 59.33, 18.06)
    gbg = _stop("GBG", "Goteborg", 57.71, 11.97)
    # A different price per day, driven by the timetable the factory hands back.
    tts = {}
    prices = {}
    for i, price in enumerate((45_000, 30_000, 80_000)):
        d = date(2026, 7, 20) + __import__("datetime").timedelta(days=i)
        tts[d] = _tt([("R", TransportMode.TRAIN, f"OP{i}", [sto, gbg], [(_hhmm(8),) * 2, (_hhmm(11),) * 2])])
        prices[f"OP{i}"] = price

    def make_planner(d):
        return _planner_for(db, tts[d], prices)

    fares = await cheapest_over_window(make_planner, "Stockholm", "Goteborg", date(2026, 7, 20), 3)
    assert [f.date for f in fares] == [date(2026, 7, 20), date(2026, 7, 21), date(2026, 7, 22)]
    prices_by_day = [f.price_ore for f in fares]
    assert prices_by_day == [45_000, 30_000, 80_000]
    # The caller finds the bargain day.
    cheapest = min(fares, key=lambda f: f.price_ore)
    assert cheapest.date == date(2026, 7, 21)


async def test_window_day_with_no_priced_option_is_none(db):
    sto = _stop("STO", "Stockholm", 59.33, 18.06)
    gbg = _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _tt([("R", TransportMode.TRAIN, "Ybuss", [sto, gbg], [(_hhmm(8),) * 2, (_hhmm(11),) * 2])])

    def make_planner(d):
        return _planner_for(db, tt, {"SJ": 1})  # nothing prices Ybuss

    fares = await cheapest_over_window(make_planner, "Stockholm", "Goteborg", date(2026, 7, 20), 2)
    assert all(f.price_ore is None for f in fares)


# --- disk-cached timetable loader ------------------------------------------


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


def test_cached_loader_writes_then_reuses_a_pickle(feed, tmp_path):
    cache = tmp_path / "ttcache"
    day = date(2026, 7, 8)

    first = load_timetable_cached(feed, day, cache_dir=cache)
    pickles = list(cache.glob("*.pkl"))
    assert len(pickles) == 1, "the first load writes one pickle"

    second = load_timetable_cached(feed, day, cache_dir=cache)
    assert second.num_stops == first.num_stops
    assert second.num_trips == first.num_trips


def test_cached_loader_matches_the_direct_parse(feed, tmp_path):
    day = date(2026, 7, 8)
    direct, _ = load_timetable(feed, day)
    cached = load_timetable_cached(feed, day, cache_dir=tmp_path / "c")
    assert cached.num_trips == direct.num_trips
    assert {s.id for s in cached.stops} == {s.id for s in direct.stops}


def test_a_new_feed_mtime_invalidates_the_cache(feed, tmp_path):
    import os
    import time

    cache = tmp_path / "c"
    day = date(2026, 7, 8)
    load_timetable_cached(feed, day, cache_dir=cache)
    # Touch the feed to a later mtime; the key changes, so a new pickle is written.
    later = time.time() + 100
    os.utime(feed, (later, later))
    load_timetable_cached(feed, day, cache_dir=cache)
    assert len(list(cache.glob("*.pkl"))) == 2


def test_no_cache_dir_falls_back_to_a_plain_parse(feed):
    tt = load_timetable_cached(feed, date(2026, 7, 8), cache_dir=None)
    assert tt.num_trips > 0
