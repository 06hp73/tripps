"""Flights: airport registry, Google Flights mapping, Swedish-only filtering, degradation."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from tripps.config import PricingBudget
from tripps.db import Database
from tripps.ingest.airports import (
    is_swedish,
    load_airports,
    nearest_airports,
    resolve_airport_stop,
)
from tripps.ingest.flights import (
    FlightOffer,
    GoogleFlightsProvider,
    NullFlightProvider,
    flight_route_addition,
)
from tripps.models import PriceConfidence, SearchConstraints, Stop, TransportMode
from tripps.pricing.flights import FlightAdapter
from tripps.pricing.orchestrator import PricingOrchestrator
from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip
from tripps.search import Planner, SearchOptions

TZ = ZoneInfo("Europe/Stockholm")


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "f.sqlite3")
    yield database
    database.close()


# --- airport registry ------------------------------------------------------


def test_registry_has_the_main_swedish_airports():
    airports = load_airports()
    for iata in ("ARN", "GOT", "MMX", "LLA", "UME", "KRN", "VBY", "BMA"):
        assert iata in airports, iata
    assert len(airports) > 25


def test_arlanda_coordinates_are_from_the_dataset_not_memory():
    arn = load_airports()["ARN"]
    assert arn.lat == pytest.approx(59.6485, abs=0.01)
    assert arn.lon == pytest.approx(17.9288, abs=0.01)


def test_only_swedish_airports_pass_the_domestic_guard():
    assert is_swedish("ARN") and is_swedish("got")
    assert not is_swedish("FRA")  # Frankfurt
    assert not is_swedish("HEL")  # Helsinki
    assert not is_swedish("CPH")  # Copenhagen, close but not Sweden


def test_nearest_airports_are_sensible():
    stockholm = nearest_airports(59.33, 18.06, limit=3)
    codes = {a.iata for a in stockholm}
    assert "ARN" in codes or "BMA" in codes
    assert "KRN" not in codes  # Kiruna does not serve Stockholm


def test_nearest_airports_respects_radius():
    assert nearest_airports(59.33, 18.06, limit=5, max_km=1.0) == []


def test_resolve_airport_stop_prefers_a_named_airport_stop():
    b = TimetableBuilder()
    arn = load_airports()["ARN"]
    b.add_stop(Stop(id="near", name="Some Village", lat=arn.lat + 0.01, lon=arn.lon))
    b.add_stop(Stop(id="apt", name="Stockholm Arlanda flygplats", lat=arn.lat + 0.02, lon=arn.lon))
    b.add_stop(Stop(id="far", name="Uppsala C", lat=59.858, lon=17.646))
    info = RouteInfo(id="R", mode=TransportMode.TRAIN, operator="SJ")
    b.add_trip(info, ["apt", "far"], Trip(id="t", arrivals=[0, 600], departures=[0, 600]))
    tt = b.build()

    stop = resolve_airport_stop(tt, arn)
    assert stop is not None and stop.id == "apt"


def test_airport_with_no_ground_access_resolves_to_nothing():
    b = TimetableBuilder()
    b.add_stop(Stop(id="a", name="Stockholm C", lat=59.33, lon=18.06))
    b.add_stop(Stop(id="b", name="Goteborg C", lat=57.71, lon=11.97))
    info = RouteInfo(id="R", mode=TransportMode.TRAIN, operator="SJ")
    b.add_trip(info, ["a", "b"], Trip(id="t", arrivals=[0, 600], departures=[0, 600]))
    tt = b.build()
    assert resolve_airport_stop(tt, load_airports()["KRN"]) is None


# --- Google Flights mapping -----------------------------------------------


def _leg(from_code, to_code, dep, arr, day: date):
    def simple(dt):
        return SimpleNamespace(date=[day.year, day.month, day.day], time=[dt[0], dt[1]])

    return SimpleNamespace(
        from_airport=SimpleNamespace(code=from_code, name=from_code),
        to_airport=SimpleNamespace(code=to_code, name=to_code),
        departure=simple(dep),
        arrival=simple(arr),
        duration=70,
        plane_type="",
    )


def _result(price, airlines, legs):
    return SimpleNamespace(type="one-way", price=price, airlines=airlines, flights=legs, carbon=None)


def test_direct_swedish_flight_is_mapped():
    day = date(2026, 8, 12)
    provider = GoogleFlightsProvider()
    results = [_result(1529, ["Scandinavian Airlines"], [_leg("ARN", "GOT", (16, 50), (18, 0), day)])]
    offers = provider._map(results, load_airports()["ARN"], load_airports()["GOT"], day)

    assert len(offers) == 1
    offer = offers[0]
    assert offer.carrier == "Scandinavian Airlines"
    assert offer.price_ore == 1529 * 100
    assert offer.departure == datetime(2026, 8, 12, 16, 50, tzinfo=TZ)
    assert offer.duration_seconds == 70 * 60


def test_foreign_connections_are_rejected():
    """A live ARN->GOT search really does return Lufthansa via Frankfurt and Finnair via
    Helsinki. Those are not journeys within Sweden."""
    day = date(2026, 8, 12)
    provider = GoogleFlightsProvider()
    results = [
        _result(1520, ["Lufthansa"], [
            _leg("ARN", "FRA", (17, 10), (19, 0), day),
            _leg("FRA", "GOT", (20, 0), (21, 30), day),
        ]),
        _result(1522, ["Finnair"], [
            _leg("ARN", "HEL", (16, 40), (18, 0), day),
            _leg("HEL", "GOT", (19, 0), (20, 30), day),
        ]),
        _result(1529, ["Scandinavian Airlines"], [_leg("ARN", "GOT", (16, 50), (18, 0), day)]),
    ]
    offers = provider._map(results, load_airports()["ARN"], load_airports()["GOT"], day)
    assert [o.carrier for o in offers] == ["Scandinavian Airlines"]


def test_single_leg_to_a_foreign_airport_is_rejected():
    day = date(2026, 8, 12)
    provider = GoogleFlightsProvider()
    results = [_result(900, ["SAS"], [_leg("ARN", "CPH", (8, 0), (9, 10), day)])]
    assert provider._map(results, load_airports()["ARN"], load_airports()["GOT"], day) == []


def test_overnight_arrival_rolls_to_the_next_day():
    day = date(2026, 8, 12)
    provider = GoogleFlightsProvider()
    results = [_result(700, ["SAS"], [_leg("ARN", "LLA", (23, 30), (0, 45), day)])]
    [offer] = provider._map(results, load_airports()["ARN"], load_airports()["LLA"], day)
    assert offer.arrival > offer.departure
    assert offer.arrival.date() == date(2026, 8, 13)


def test_malformed_results_are_skipped():
    day = date(2026, 8, 12)
    provider = GoogleFlightsProvider()
    results = [
        _result(None, ["SAS"], [_leg("ARN", "GOT", (8, 0), (9, 10), day)]),
        _result(0, ["SAS"], [_leg("ARN", "GOT", (8, 0), (9, 10), day)]),
        _result(500, [], []),
    ]
    assert provider._map(results, load_airports()["ARN"], load_airports()["GOT"], day) == []


# --- synthetic flight routes ----------------------------------------------


def _offer(dep_hour, price, day=date(2026, 8, 12), carrier="SAS") -> FlightOffer:
    dep = datetime(day.year, day.month, day.day, dep_hour, 0, tzinfo=TZ)
    return FlightOffer(
        carrier=carrier,
        from_iata="ARN",
        to_iata="GOT",
        departure=dep,
        arrival=dep + timedelta(minutes=70),
        price_ore=price,
    )


def test_flight_route_carries_the_exact_fare_on_each_trip():
    day = date(2026, 8, 12)
    a = Stop(id="apt-a", name="Arlanda", lat=59.65, lon=17.93)
    b = Stop(id="apt-b", name="Landvetter", lat=57.66, lon=12.28)
    addition = flight_route_addition([_offer(8, 89_900), _offer(17, 152_900)], day, a, b)

    assert addition is not None
    assert addition.info.mode is TransportMode.FLIGHT
    assert addition.info.synthetic
    fares = sorted(t.precomputed_fare_ore for t in addition.trips)
    assert fares == [89_900, 152_900]


def test_flight_route_needs_at_least_one_valid_trip():
    day = date(2026, 8, 12)
    a = Stop(id="apt-a", name="Arlanda", lat=59.65, lon=17.93)
    b = Stop(id="apt-b", name="Landvetter", lat=57.66, lon=12.28)
    assert flight_route_addition([], day, a, b) is None


# --- flight price adapter --------------------------------------------------


async def test_flight_adapter_quotes_the_offer_it_was_given():
    offer = _offer(8, 89_900)
    adapter = FlightAdapter([offer])
    from tripps.models import Leg

    leg = Leg(
        from_stop=Stop(id="apt-a", name="Arlanda", lat=59.65, lon=17.93),
        to_stop=Stop(id="apt-b", name="Landvetter GOT", lat=57.66, lon=12.28),
        mode=TransportMode.FLIGHT,
        operator="air",
        departure=offer.departure,
        arrival=offer.arrival,
        service_ref=offer.route_id,
    )
    quote = await adapter.quote_leg(leg)
    assert quote.confidence is PriceConfidence.EXACT
    assert quote.amount_ore == 89_900
    assert "google flights" in (quote.note or "").lower()
    assert quote.deeplink


async def test_flight_adapter_reports_missing_offer():
    from tripps.models import Leg

    adapter = FlightAdapter([])
    leg = Leg(
        from_stop=Stop(id="a", name="Arlanda", lat=59.65, lon=17.93),
        to_stop=Stop(id="b", name="Landvetter", lat=57.66, lon=12.28),
        mode=TransportMode.FLIGHT,
        operator="air",
        departure=datetime(2026, 8, 12, 8, 0, tzinfo=TZ),
        arrival=datetime(2026, 8, 12, 9, 10, tzinfo=TZ),
        service_ref="air:ARN-GOT:0800",
    )
    quote = await adapter.quote_leg(leg)
    assert quote.confidence is PriceConfidence.UNAVAILABLE


async def test_flight_adapter_makes_no_http_call_of_its_own():
    """The search already paid for the page scrape; phase 2 reads it back."""
    import respx

    from tripps.models import Leg

    offer = _offer(8, 89_900)
    with respx.mock(assert_all_called=False) as mock:
        adapter = FlightAdapter([offer])
        leg = Leg(
            from_stop=Stop(id="a", name="Arlanda", lat=59.65, lon=17.93),
            to_stop=Stop(id="b", name="Landvetter GOT", lat=57.66, lon=12.28),
            mode=TransportMode.FLIGHT,
            operator="air",
            departure=offer.departure,
            arrival=offer.arrival,
            service_ref=offer.route_id,
        )
        await adapter.quote_leg(leg)
        assert not mock.calls


# --- integration: a broken scraper must not break the search ----------------


class BrokenProvider:
    async def search(self, origin, destination, day):
        raise RuntimeError("Google returned its consent page")


class StubProvider:
    def __init__(self, offers):
        self.offers = offers
        self.calls = 0

    async def search(self, origin, destination, day):
        self.calls += 1
        if (origin.iata, destination.iata) != ("ARN", "GOT"):
            return []
        return self.offers


def _network_with_airports():
    arn, got = load_airports()["ARN"], load_airports()["GOT"]
    b = TimetableBuilder()
    b.add_stop(Stop(id="STO", name="Stockholm C", lat=59.33, lon=18.06))
    b.add_stop(Stop(id="ARN_S", name="Stockholm Arlanda flygplats", lat=arn.lat, lon=arn.lon))
    b.add_stop(Stop(id="GOT_S", name="Landvetter flygplats", lat=got.lat, lon=got.lon))
    b.add_stop(Stop(id="GBG", name="Goteborg C", lat=57.71, lon=11.97))
    day = date(2026, 8, 12)

    # Airport coaches at both ends, so a flight is actually reachable on the ground.
    train = RouteInfo(id="ARL", mode=TransportMode.TRAIN, operator="SJ")
    b.add_trip(train, ["STO", "ARN_S"],
               Trip(id="arl", arrivals=[6 * 3600, 6 * 3600 + 1200], departures=[6 * 3600, 6 * 3600 + 1200]))
    coach = RouteInfo(id="FLY", mode=TransportMode.BUS, operator="Flygbussarna")
    b.add_trip(coach, ["GOT_S", "GBG"],
               Trip(id="fly", arrivals=[10 * 3600, 10 * 3600 + 1800], departures=[10 * 3600, 10 * 3600 + 1800]))
    # A slow direct train, so there is always something to fall back to.
    slow = RouteInfo(id="SLOW", mode=TransportMode.TRAIN, operator="SJ")
    b.add_trip(slow, ["STO", "GBG"],
               Trip(id="slow", arrivals=[6 * 3600, 14 * 3600], departures=[6 * 3600, 14 * 3600]))
    b.add_transfer("STO", "STO", 0)
    return b.build(), day


def _planner(tt, provider, db, adapters):
    orch = PricingOrchestrator(adapters, db, budget=PricingBudget(min_interval_seconds=0.0))
    return Planner(tt, orch, db=db, flight_provider=provider,
                   options=SearchOptions(include_freerider=False))


async def test_broken_flight_provider_degrades_with_a_warning(db):
    tt, day = _network_with_airports()
    planner = _planner(tt, BrokenProvider(), db, [FlightAdapter()])
    response, stats = await planner.search("Stockholm C", "Goteborg C", day)

    assert stats.flight_offers == 0
    assert response.itineraries, "the ground journey must still be returned"
    assert any("unavailable" in w.lower() for w in response.warnings)


async def test_flight_leg_is_routed_and_priced(db):
    tt, day = _network_with_airports()
    dep = datetime(day.year, day.month, day.day, 8, 0, tzinfo=TZ)
    offer = FlightOffer(
        carrier="SAS", from_iata="ARN", to_iata="GOT",
        departure=dep, arrival=dep + timedelta(minutes=70), price_ore=59_900,
    )
    planner = _planner(tt, StubProvider([offer]), db, [FlightAdapter()])
    response, stats = await planner.search("Stockholm C", "Goteborg C", day)

    assert stats.flight_offers == 1
    flying = [i for i in response.itineraries if TransportMode.FLIGHT in i.modes]
    assert flying, "a train-to-airport + flight + coach itinerary must be found"

    itin = flying[0]
    assert itin.is_multimodal
    flight_leg = next(leg for leg in itin.legs if leg.mode is TransportMode.FLIGHT)
    assert flight_leg.quote.amount_ore == 59_900
    assert flight_leg.quote.confidence is PriceConfidence.EXACT


async def test_flight_offers_are_cached_between_searches(db):
    tt, day = _network_with_airports()
    dep = datetime(day.year, day.month, day.day, 8, 0, tzinfo=TZ)
    offer = FlightOffer(
        carrier="SAS", from_iata="ARN", to_iata="GOT",
        departure=dep, arrival=dep + timedelta(minutes=70), price_ore=59_900,
    )
    provider = StubProvider([offer])
    planner = _planner(tt, provider, db, [FlightAdapter()])

    await planner.search("Stockholm C", "Goteborg C", day)
    first = provider.calls
    assert first > 0

    await planner.search("Stockholm C", "Goteborg C", day)
    assert provider.calls == first, "a page scrape must not repeat within its TTL"


async def test_excluding_flights_skips_the_scraper_entirely(db):
    tt, day = _network_with_airports()
    provider = StubProvider([])
    planner = _planner(tt, provider, db, [FlightAdapter()])
    modes = frozenset(TransportMode) - {TransportMode.FLIGHT}

    response, stats = await planner.search(
        "Stockholm C", "Goteborg C", day, constraints=SearchConstraints(allowed_modes=modes)
    )
    assert provider.calls == 0
    assert stats.flight_offers == 0
    assert all(TransportMode.FLIGHT not in i.modes for i in response.itineraries)


async def test_null_provider_returns_nothing():
    provider = NullFlightProvider()
    assert await provider.search(load_airports()["ARN"], load_airports()["GOT"], date(2026, 8, 12)) == []
