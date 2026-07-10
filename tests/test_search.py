"""End-to-end: route on schedules, price the survivors, rank by real cost.

The headline requirement of the project is a mixed-mode itinerary (train leg 1, Hertz
Freerider leg 2). `test_mixed_mode_train_then_freerider_wins_on_price` is that requirement
expressed as a test, over a real Freerider offer from the recorded snapshot.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tripps.config import CacheTTL, PricingBudget
from tripps.db import Database
from tripps.ingest.freerider import FreeriderCostModel, parse_offers
from tripps.interfaces import HealthState, PriceAdapter, SourceHealth
from tripps.models import (
    Leg,
    PriceConfidence,
    Quote,
    SearchConstraints,
    Stop,
    TransportMode,
)
from tripps.pricing.freerider import FreeriderAdapter
from tripps.pricing.orchestrator import PricingOrchestrator
from tripps.routing.floors import DEFAULT_FLOORS, ModeFloor, PriceFloorModel
from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip
from tripps.search import Planner, SearchOptions, summarize
from tripps.timeutil import to_service_seconds

TZ = ZoneInfo("Europe/Stockholm")
FIXTURES = Path(__file__).parent / "fixtures"


class StubAdapter(PriceAdapter):
    """A price source with fixed answers, so ranking can be tested deterministically."""

    modes = frozenset({TransportMode.TRAIN, TransportMode.BUS})

    def __init__(self, name: str, prices: dict[str, int], *, fail: bool = False) -> None:
        self.name = name
        self.prices = prices
        self.fail = fail
        self.calls = 0

    def supports(self, leg: Leg) -> bool:
        return leg.mode in self.modes and (leg.operator or "") in self.prices

    async def quote_leg(self, leg: Leg) -> Quote:
        self.calls += 1
        if self.fail:
            raise RuntimeError("upstream exploded")
        return Quote(
            source=self.name,
            amount_ore=self.prices[leg.operator],
            confidence=PriceConfidence.EXACT,
            fetched_at=datetime.now(tz=TZ),
        )

    async def health(self) -> SourceHealth:
        return SourceHealth(self.name, HealthState.DOWN if self.fail else HealthState.OK)


@pytest.fixture
def offers():
    raw = json.loads((FIXTURES / "freerider_sweden.json").read_text(encoding="utf-8"))
    return parse_offers(raw)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "t.sqlite3")
    yield database
    database.close()


def _stop(sid, name, lat, lon) -> Stop:
    return Stop(id=sid, name=name, lat=lat, lon=lon)


def _timetable_with(routes):
    b = TimetableBuilder()
    for route_id, mode, operator, stops, times in routes:
        for stop in stops:
            b.add_stop(stop)
        info = RouteInfo(id=route_id, mode=mode, operator=operator)
        b.add_trip(
            info,
            [s.id for s in stops],
            Trip(id=f"{route_id}-t", arrivals=[t for t, _ in times], departures=[d for _, d in times]),
        )
    return b.build()


def _hhmm(h, m=0) -> int:
    return h * 3600 + m * 60


def _planner(timetable, adapters, db=None, **kw) -> Planner:
    orchestrator = PricingOrchestrator(
        adapters,
        db,
        budget=kw.pop("budget", PricingBudget(min_interval_seconds=0.0)),
        ttl=kw.pop("ttl", CacheTTL()),
        floors=kw.pop("floors", PriceFloorModel()),
    )
    return Planner(
        timetable,
        orchestrator,
        floors=kw.pop("floors_router", PriceFloorModel()),
        db=db,
        options=kw.pop("options", SearchOptions()),
    )


# --- ranking ---------------------------------------------------------------


async def test_cheapest_wins_even_when_slower(db):
    """The point of the whole system: a slow cheap bus must outrank a fast dear train."""
    sto, gbg = _stop("STO", "Stockholm", 59.33, 18.06), _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _timetable_with(
        [
            ("FAST", TransportMode.TRAIN, "SJ", [sto, gbg], [(_hhmm(8), _hhmm(8)), (_hhmm(11), _hhmm(11))]),
            ("SLOW", TransportMode.BUS, "FlixBus", [sto, gbg], [(_hhmm(8), _hhmm(8)), (_hhmm(15), _hhmm(15))]),
        ]
    )
    adapter = StubAdapter("stub", {"SJ": 89_500, "FlixBus": 19_900})
    planner = _planner(tt, [adapter], db)

    response, stats = await planner.search("Stockholm", "Goteborg", date(2026, 7, 8))

    assert len(response.itineraries) == 2
    best = response.itineraries[0]
    assert best.legs[0].operator == "FlixBus"
    assert best.total_price_ore == 19_900
    assert best.total_price_sek == 199.0
    assert stats.candidates == 2


async def test_unpriced_itinerary_never_outranks_a_priced_one(db):
    """A missing leg price must not read as free."""
    sto, gbg = _stop("STO", "Stockholm", 59.33, 18.06), _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _timetable_with(
        [
            ("A", TransportMode.TRAIN, "SJ", [sto, gbg], [(_hhmm(8), _hhmm(8)), (_hhmm(11), _hhmm(11))]),
            ("B", TransportMode.BUS, "Ybuss", [sto, gbg], [(_hhmm(8), _hhmm(8)), (_hhmm(16), _hhmm(16))]),
        ]
    )
    # Only SJ is priceable; Ybuss has no adapter at all.
    planner = _planner(tt, [StubAdapter("stub", {"SJ": 89_500})], db)
    response, _ = await planner.search("Stockholm", "Goteborg", date(2026, 7, 8))

    assert response.itineraries[0].total_price_ore == 89_500
    unpriced = response.itineraries[1]
    assert unpriced.total_price_ore is None
    assert not unpriced.fully_priced
    assert unpriced.price_confidence is PriceConfidence.UNAVAILABLE
    assert any("Could not price" in w for w in response.warnings)


async def test_adapter_exception_does_not_break_the_search(db):
    sto, gbg = _stop("STO", "Stockholm", 59.33, 18.06), _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _timetable_with(
        [("A", TransportMode.TRAIN, "SJ", [sto, gbg], [(_hhmm(8), _hhmm(8)), (_hhmm(11), _hhmm(11))])]
    )
    planner = _planner(tt, [StubAdapter("boom", {"SJ": 1}, fail=True)], db)
    response, _ = await planner.search("Stockholm", "Goteborg", date(2026, 7, 8))

    assert len(response.itineraries) == 1
    assert response.itineraries[0].total_price_ore is None
    assert response.source_status["boom"] == "down"


# --- the headline requirement ---------------------------------------------


async def test_mixed_mode_train_then_freerider_wins_on_price(db, offers):
    """Train from Stockholm to Uppsala, then a free Hertz car to Arlanda.

    Uses a real offer from the recorded Freerider snapshot (Uppsala -> Stockholm Arlanda,
    33 km, free). The direct train is expensive; the cheap feeder train plus the free car
    undercuts it. That is the entire value proposition of the project.
    """
    offer = next(o for o in offers if o.pickup.city == "Uppsala")
    service_date = offer.available_at.date()
    pickup = offer.pickup.to_stop()
    dropoff = offer.dropoff.to_stop()

    # A GTFS-ish stop right next to the Freerider pickup station, so the car is walkable.
    sto = _stop("STO", "Stockholm", 59.33, 18.06)
    uppsala = Stop(
        id="UPP", name="Uppsala C", lat=pickup.lat + 0.002, lon=pickup.lon + 0.002
    )
    arlanda = Stop(id="ARN", name="Arlanda", lat=dropoff.lat, lon=dropoff.lon)

    depart = offer.available_at - timedelta(hours=2)
    arrive = offer.available_at - timedelta(minutes=30)

    tt = _timetable_with(
        [
            (
                "TRAIN",
                TransportMode.TRAIN,
                "SJ",
                [sto, uppsala],
                [
                    (to_service_seconds(depart, service_date),) * 2,
                    (to_service_seconds(arrive, service_date),) * 2,
                ],
            ),
            (
                "DIRECT",
                TransportMode.TRAIN,
                "Arlanda Express",
                [sto, arlanda],
                [
                    (to_service_seconds(depart, service_date),) * 2,
                    (to_service_seconds(offer.available_at + timedelta(hours=1), service_date),) * 2,
                ],
            ),
        ]
    )

    adapters = [
        # Cheap regional feeder to Uppsala; expensive direct service to the airport.
        StubAdapter("rail", {"SJ": 15_000, "Arlanda Express": 32_000}),
        FreeriderAdapter(cost_model=FreeriderCostModel()),
    ]
    planner = _planner(tt, adapters, db, options=SearchOptions(freerider_step_minutes=15))

    response, stats = await planner.search(
        "Stockholm",
        "Arlanda",
        service_date,
        offers=[offer],
        now=depart - timedelta(hours=1),
    )

    assert stats.freerider_offers == 1, "the car must be overlaid onto the network"

    mixed = [i for i in response.itineraries if TransportMode.FREERIDER in i.modes]
    assert mixed, "a train+Freerider itinerary must be found"

    best = response.itineraries[0]
    assert best.is_multimodal, "the cheapest journey combines a train and a free car"
    assert best.modes[0] is TransportMode.TRAIN
    assert TransportMode.FREERIDER in best.modes
    # 150 SEK feeder + free car beats the 320 SEK direct train.
    assert best.total_price_ore == 15_000

    direct = next(i for i in response.itineraries if TransportMode.FREERIDER not in i.modes)
    assert direct.total_price_ore == 32_000
    assert best.total_price_ore < direct.total_price_ore

    car_leg = next(leg for leg in best.legs if leg.mode is TransportMode.FREERIDER)
    assert car_leg.quote.amount_ore == 0
    assert car_leg.quote.confidence is PriceConfidence.ESTIMATED, (
        "a Freerider price is our cost model's assumption, never a Hertz quote"
    )

    blob = " ".join(response.warnings).lower()
    assert "cannot be reserved from here" in blob
    assert "return by" in blob


async def test_excluding_freerider_removes_car_legs(db, offers):
    offer = next(o for o in offers if o.pickup.city == "Uppsala")
    service_date = offer.available_at.date()
    pickup, dropoff = offer.pickup.to_stop(), offer.dropoff.to_stop()
    sto = _stop("STO", "Stockholm", 59.33, 18.06)
    uppsala = Stop(id="UPP", name="Uppsala C", lat=pickup.lat + 0.002, lon=pickup.lon + 0.002)
    arlanda = Stop(id="ARN", name="Arlanda", lat=dropoff.lat, lon=dropoff.lon)
    depart = offer.available_at - timedelta(hours=2)

    tt = _timetable_with(
        [
            ("T", TransportMode.TRAIN, "SJ", [sto, uppsala],
             [(to_service_seconds(depart, service_date),) * 2,
              (to_service_seconds(offer.available_at - timedelta(minutes=30), service_date),) * 2]),
            ("D", TransportMode.TRAIN, "SJ", [sto, arlanda],
             [(to_service_seconds(depart, service_date),) * 2,
              (to_service_seconds(offer.available_at, service_date),) * 2]),
        ]
    )
    planner = _planner(tt, [StubAdapter("rail", {"SJ": 15_000}), FreeriderAdapter()], db)

    response, stats = await planner.search(
        "Stockholm",
        "Arlanda",
        service_date,
        constraints=SearchConstraints(include_freerider=False),
        offers=[offer],
        now=depart - timedelta(hours=1),
    )
    assert stats.freerider_offers == 0
    assert all(TransportMode.FREERIDER not in i.modes for i in response.itineraries)


# --- constraints -----------------------------------------------------------


async def test_max_transfers_constraint(db):
    sto = _stop("STO", "Stockholm", 59.33, 18.06)
    nrk = _stop("NRK", "Norrkoping", 58.596, 16.183)
    gbg = _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _timetable_with(
        [
            ("L1", TransportMode.TRAIN, "SJ", [sto, nrk], [(_hhmm(8),) * 2, (_hhmm(9),) * 2]),
            ("L2", TransportMode.TRAIN, "SJ", [nrk, gbg], [(_hhmm(10),) * 2, (_hhmm(13),) * 2]),
        ]
    )
    planner = _planner(tt, [StubAdapter("rail", {"SJ": 10_000})], db)

    allowed, _ = await planner.search("Stockholm", "Goteborg", date(2026, 7, 8))
    assert len(allowed.itineraries) == 1

    blocked, _ = await planner.search(
        "Stockholm", "Goteborg", date(2026, 7, 8),
        constraints=SearchConstraints(max_transfers=0),
    )
    assert blocked.itineraries == []
    assert any("constraints" in w for w in blocked.warnings)


async def test_max_duration_constraint(db):
    sto, gbg = _stop("STO", "Stockholm", 59.33, 18.06), _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _timetable_with(
        [("A", TransportMode.TRAIN, "SJ", [sto, gbg], [(_hhmm(8),) * 2, (_hhmm(16),) * 2])]
    )
    planner = _planner(tt, [StubAdapter("rail", {"SJ": 10_000})], db)
    response, _ = await planner.search(
        "Stockholm", "Goteborg", date(2026, 7, 8),
        constraints=SearchConstraints(max_duration_seconds=4 * 3600),
    )
    assert response.itineraries == []


async def test_unknown_place_raises(db):
    tt = _timetable_with(
        [("A", TransportMode.TRAIN, "SJ",
          [_stop("STO", "Stockholm", 59.33, 18.06), _stop("GBG", "Goteborg", 57.71, 11.97)],
          [(_hhmm(8),) * 2, (_hhmm(11),) * 2])]
    )
    planner = _planner(tt, [], db)
    with pytest.raises(LookupError, match="Atlantis"):
        await planner.search("Atlantis", "Goteborg", date(2026, 7, 8))


# --- cache and budget ------------------------------------------------------


async def test_second_search_is_served_from_cache(db):
    sto, gbg = _stop("STO", "Stockholm", 59.33, 18.06), _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _timetable_with(
        [("A", TransportMode.TRAIN, "SJ", [sto, gbg], [(_hhmm(8),) * 2, (_hhmm(11),) * 2])]
    )
    adapter = StubAdapter("rail", {"SJ": 10_000})
    planner = _planner(tt, [adapter], db)

    first, _ = await planner.search("Stockholm", "Goteborg", date(2026, 7, 8))
    assert adapter.calls == 1
    assert first.itineraries[0].price_confidence is PriceConfidence.EXACT

    second, _ = await planner.search("Stockholm", "Goteborg", date(2026, 7, 8))
    assert adapter.calls == 1, "the second search must not hit upstream"
    assert second.itineraries[0].price_confidence is PriceConfidence.CACHED
    assert second.itineraries[0].total_price_ore == 10_000


async def test_call_budget_caps_real_http_requests(db):
    """The budget is charged where a request leaves the process, not per leg priced."""
    import httpx
    import respx

    from tripps.pricing.base import CallBudget
    from tripps.pricing.flixbus import DEFAULT_BASE, FlixBusAdapter

    with respx.mock:
        cities = respx.get(url__startswith=f"{DEFAULT_BASE}/search/autocomplete/cities").mock(
            return_value=httpx.Response(200, json=[{"id": "c1", "name": "X"}])
        )
        adapter = FlixBusAdapter(min_interval=0.0)
        adapter.set_budget(CallBudget(limit=2))

        results = []
        for _ in range(5):
            try:
                results.append(await adapter.resolve_city(f"place-{len(results)}"))
            except Exception as exc:  # BudgetExceeded
                results.append(exc)
        await adapter.aclose()

    assert cities.call_count == 2, "the third request must never leave the process"


async def test_one_flixbus_response_prices_many_departures(db):
    """Pricing six departures of the same city pair is one HTTP request, not six.

    This is why the candidate cap can be generous: sampling the whole day's price curve
    costs the same as sampling one departure.
    """
    import httpx
    import respx

    from tripps.pricing.flixbus import DEFAULT_BASE, FlixBusAdapter

    payload = {
        "trips": [
            {
                "results": {
                    f"u{h}": {
                        "uid": f"u{h}",
                        "price": {"total": 400 + h, "total_with_platform_fee": 411 + h},
                        "departure": {"date": f"2026-07-13T{h:02d}:30:00+02:00"},
                        "arrival": {"date": f"2026-07-13T{h + 6:02d}:45:00+02:00"},
                        "status": "available",
                    }
                    for h in (7, 10, 15)
                }
            }
        ]
    }
    with respx.mock:
        respx.get(url__startswith=f"{DEFAULT_BASE}/search/autocomplete/cities").mock(
            return_value=httpx.Response(200, json=[{"id": "c1", "name": "X"}])
        )
        search = respx.get(url__startswith=f"{DEFAULT_BASE}/search/service/v4/search").mock(
            return_value=httpx.Response(200, json=payload)
        )
        adapter = FlixBusAdapter(min_interval=0.0)
        quotes = []
        for hour in (7, 10, 15):
            leg = Leg(
                from_stop=_stop("STO", "Stockholm", 59.33, 18.06),
                to_stop=_stop("GBG", "Goteborg", 57.71, 11.97),
                mode=TransportMode.BUS,
                operator="FlixBus",
                departure=datetime(2026, 7, 13, hour, 30, tzinfo=TZ),
                arrival=datetime(2026, 7, 13, hour + 6, 45, tzinfo=TZ),
            )
            quotes.append(await adapter.quote_leg(leg))
        await adapter.aclose()

    assert search.call_count == 1
    assert [q.amount_ore for q in quotes] == [41_800, 42_100, 42_600]
    assert all(q.confidence is PriceConfidence.EXACT for q in quotes)


async def test_floor_violation_is_detected_and_warned(db):
    """If the routing floor exceeds a real fare, the cheapest journey may have been pruned.
    That must surface, not hide."""
    sto, gbg = _stop("STO", "Stockholm", 59.33, 18.06), _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _timetable_with(
        [("A", TransportMode.TRAIN, "SJ", [sto, gbg], [(_hhmm(8),) * 2, (_hhmm(11),) * 2])]
    )
    # A deliberately broken floor: 900 SEK minimum for any SJ train.
    broken = PriceFloorModel(DEFAULT_FLOORS, {"SJ": ModeFloor(base_ore=90_000, per_km_ore=0)})
    adapter = StubAdapter("rail", {"SJ": 19_900})  # the real fare is 199 SEK
    planner = _planner(tt, [adapter], db, floors=broken, floors_router=broken)

    response, _ = await planner.search("Stockholm", "Goteborg", date(2026, 7, 8))
    assert any("price-floor violation" in w for w in response.warnings)
    assert db.floor_violations(), "the violation must be persisted for calibration"


async def test_reprice_deltas_are_recorded_for_calibration(db):
    sto, gbg = _stop("STO", "Stockholm", 59.33, 18.06), _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _timetable_with(
        [("A", TransportMode.TRAIN, "SJ", [sto, gbg], [(_hhmm(8),) * 2, (_hhmm(11),) * 2])]
    )
    planner = _planner(tt, [StubAdapter("rail", {"SJ": 50_000})], db)
    await planner.search("Stockholm", "Goteborg", date(2026, 7, 8))

    rows = db._conn.execute("SELECT * FROM reprice_delta").fetchall()  # noqa: SLF001
    assert len(rows) == 1
    assert rows[0]["actual_ore"] == 50_000
    assert rows[0]["floor_ore"] < rows[0]["actual_ore"]
    assert rows[0]["distance_km"] > 300  # Stockholm-Goteborg


# --- presentation ----------------------------------------------------------


async def test_summarize_reads_sensibly(db):
    sto, gbg = _stop("STO", "Stockholm", 59.33, 18.06), _stop("GBG", "Goteborg", 57.71, 11.97)
    tt = _timetable_with(
        [("A", TransportMode.BUS, "FlixBus", [sto, gbg], [(_hhmm(8),) * 2, (_hhmm(14), _hhmm(14))])]
    )
    planner = _planner(tt, [StubAdapter("stub", {"FlixBus": 19_900})], db)
    response, _ = await planner.search("Stockholm", "Goteborg", date(2026, 7, 8))
    line = summarize(response.itineraries[0])
    assert "08:00-14:00" in line
    assert "6h00m" in line
    assert "199 SEK" in line
    assert "bus" in line
