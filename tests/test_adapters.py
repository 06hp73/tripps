"""Price adapters: schema contracts against recorded live responses, and failure behaviour.

The FlixBus fixtures are a real capture of `global.api.flixbus.com` from 2026-07-10. If the
endpoint changes shape, these tests fail here rather than silently mispricing a leg.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx

from tripps.ingest.freerider import parse_offers
from tripps.models import Leg, PriceConfidence, Stop, TransportMode
from tripps.pricing.base import BudgetExceeded, CallBudget, ore_from_major
from tripps.pricing.flixbus import (
    DEFAULT_BASE,
    FlixBusAdapter,
    format_flix_date,
    is_flixbus,
    parse_search,
)
from tripps.pricing.freerider import FreeriderAdapter

TZ = ZoneInfo("Europe/Stockholm")
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def flix_search() -> dict:
    return json.loads((FIXTURES / "flixbus_search.json").read_text(encoding="utf-8"))


@pytest.fixture
def flix_cities() -> list:
    return json.loads((FIXTURES / "flixbus_cities.json").read_text(encoding="utf-8"))


def _stop(sid: str, name: str, lat: float, lon: float, **ext) -> Stop:
    return Stop(id=sid, name=name, lat=lat, lon=lon, external_ids=ext)


def _flix_leg(dep: datetime, arr: datetime, operator: str = "FlixBus") -> Leg:
    return Leg(
        from_stop=_stop("STO", "Stockholm", 59.33, 18.06),
        to_stop=_stop("GBG", "Goteborg", 57.71, 11.97),
        mode=TransportMode.BUS,
        operator=operator,
        departure=dep,
        arrival=arr,
        service_ref="T1",
    )


# --- money -----------------------------------------------------------------


def test_ore_conversion_rounds_half_up():
    assert ore_from_major(439) == 43_900
    assert ore_from_major("43.99") == 4_399  # binary float would truncate to 4398
    assert ore_from_major(0) == 0


def test_negative_amount_rejected():
    with pytest.raises(ValueError, match="negative"):
        ore_from_major(-1)


# --- FlixBus parsing -------------------------------------------------------


def test_flix_date_format_is_dotted_not_iso():
    """An ISO date makes the endpoint return HTTP 400."""
    assert format_flix_date(date(2026, 7, 24)) == "24.07.2026"


def test_operator_detection():
    assert is_flixbus("FlixBus") and is_flixbus("flixbus") and is_flixbus("Swebus")
    assert not is_flixbus("SJ") and not is_flixbus(None)


def test_parse_search_reads_the_live_schema(flix_search):
    departures = parse_search(flix_search)
    assert len(departures) == 3
    first = departures[0]
    assert first.departure == datetime(2026, 7, 24, 7, 30, tzinfo=TZ)
    assert first.status == "available" and first.bookable
    assert first.seats_left == 42


def test_platform_fee_is_included_in_the_quoted_total(flix_search):
    """`total` omits the booking fee the passenger actually pays."""
    departures = parse_search(flix_search)
    cheap = min(departures, key=lambda d: d.total_with_fee_ore)
    assert cheap.total_ore == 408 * 100
    assert cheap.total_with_fee_ore == 419 * 100
    assert cheap.total_with_fee_ore > cheap.total_ore


def test_parse_search_skips_malformed_results():
    payload = {"trips": [{"results": {"a": {"price": {}}, "b": {"nonsense": 1}}}]}
    assert parse_search(payload) == []


def test_parse_search_tolerates_missing_platform_fee():
    payload = {
        "trips": [
            {
                "results": {
                    "x": {
                        "uid": "x",
                        "price": {"total": 100},
                        "departure": {"date": "2026-07-24T07:30:00+02:00"},
                        "arrival": {"date": "2026-07-24T13:45:00+02:00"},
                        "status": "available",
                    }
                }
            }
        ]
    }
    [dep] = parse_search(payload)
    assert dep.total_with_fee_ore == dep.total_ore == 10_000


def test_parse_search_tolerates_null_platform_fee_and_keeps_other_departures():
    """A PRESENT-but-null fee is not covered by .get's default; it used to escape as a
    TypeError from ore_from_major OUTSIDE the guard, aborting the whole day's parse (and
    quote_leg's httpx/ValueError net does not catch TypeError, so the leg went unpriced)."""
    good = {
        "uid": "good",
        "price": {"total": 200, "total_with_platform_fee": 211},
        "departure": {"date": "2026-07-24T09:30:00+02:00"},
        "arrival": {"date": "2026-07-24T15:45:00+02:00"},
        "status": "available",
    }
    nulled = {
        "uid": "nulled",
        "price": {"total": 100, "total_with_platform_fee": None},
        "departure": {"date": "2026-07-24T07:30:00+02:00"},
        "arrival": {"date": "2026-07-24T13:45:00+02:00"},
        "status": "available",
    }
    payload = {"trips": [{"results": {"good": good, "nulled": nulled}}]}
    departures = parse_search(payload)
    assert len(departures) == 2, "a null fee must not abort the parse"
    by_uid = {d.uid: d for d in departures}
    assert by_uid["nulled"].total_with_fee_ore == 10_000  # null collapses to the plain total
    assert by_uid["good"].total_with_fee_ore == 21_100


# --- FlixBus adapter over mocked HTTP --------------------------------------


def _mock_flix(respx_mock, flix_cities, flix_search):
    respx_mock.get(url__startswith=f"{DEFAULT_BASE}/search/autocomplete/cities").mock(
        return_value=httpx.Response(200, json=flix_cities)
    )
    respx_mock.get(url__startswith=f"{DEFAULT_BASE}/search/service/v4/search").mock(
        return_value=httpx.Response(200, json=flix_search)
    )


@respx.mock
async def test_matching_departure_yields_exact_quote(flix_cities, flix_search):
    _mock_flix(respx.mock, flix_cities, flix_search)
    adapter = FlixBusAdapter(min_interval=0.0)
    leg = _flix_leg(
        datetime(2026, 7, 24, 7, 30, tzinfo=TZ), datetime(2026, 7, 24, 13, 45, tzinfo=TZ)
    )
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()

    assert quote.confidence is PriceConfidence.EXACT
    assert quote.amount_ore == 450 * 100  # the 13:45 arrival, incl. booking fee
    assert quote.deeplink and "flixbus" in quote.deeplink


@respx.mock
async def test_arrival_time_disambiguates_two_buses_leaving_at_once(flix_cities, flix_search):
    """The live response has two 07:30 departures at different prices. Departure time alone
    would pick one at random."""
    _mock_flix(respx.mock, flix_cities, flix_search)
    adapter = FlixBusAdapter(min_interval=0.0)
    leg = _flix_leg(
        datetime(2026, 7, 24, 7, 30, tzinfo=TZ), datetime(2026, 7, 24, 15, 25, tzinfo=TZ)
    )
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()

    assert quote.amount_ore == 419 * 100  # the slower, cheaper 15:25 arrival


@respx.mock
async def test_departure_outside_tolerance_falls_back_to_cheapest_and_flags_it(
    flix_cities, flix_search
):
    _mock_flix(respx.mock, flix_cities, flix_search)
    adapter = FlixBusAdapter(min_interval=0.0)
    leg = _flix_leg(
        datetime(2026, 7, 24, 19, 0, tzinfo=TZ), datetime(2026, 7, 24, 23, 0, tzinfo=TZ)
    )
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()

    assert quote.confidence is PriceConfidence.ESTIMATED
    assert quote.amount_ore == 419 * 100
    assert "cheapest departure" in (quote.note or "")


@respx.mock
async def test_small_timetable_drift_still_matches(flix_cities, flix_search):
    """GTFS and the sales system drift by a minute or two; that is the same bus."""
    _mock_flix(respx.mock, flix_cities, flix_search)
    adapter = FlixBusAdapter(min_interval=0.0)
    leg = _flix_leg(
        datetime(2026, 7, 24, 7, 32, tzinfo=TZ), datetime(2026, 7, 24, 13, 44, tzinfo=TZ)
    )
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.EXACT


@respx.mock
async def test_upstream_failure_degrades_to_unavailable_with_deeplink(flix_cities):
    respx.get(url__startswith=f"{DEFAULT_BASE}/search/autocomplete/cities").mock(
        return_value=httpx.Response(200, json=flix_cities)
    )
    respx.get(url__startswith=f"{DEFAULT_BASE}/search/service/v4/search").mock(
        return_value=httpx.Response(503)
    )
    adapter = FlixBusAdapter(min_interval=0.0)
    leg = _flix_leg(
        datetime(2026, 7, 24, 7, 30, tzinfo=TZ), datetime(2026, 7, 24, 13, 45, tzinfo=TZ)
    )
    quote = await adapter.quote_leg(leg)
    health = await adapter.health()
    await adapter.aclose()

    assert quote.confidence is PriceConfidence.UNAVAILABLE
    assert quote.amount_ore is None
    assert quote.deeplink  # the user can still go and book it themselves
    assert health.state.value == "down"


@respx.mock
async def test_unresolvable_city_does_not_raise(flix_search):
    respx.get(url__startswith=f"{DEFAULT_BASE}/search/autocomplete/cities").mock(
        return_value=httpx.Response(200, json=[])
    )
    adapter = FlixBusAdapter(min_interval=0.0)
    leg = _flix_leg(
        datetime(2026, 7, 24, 7, 30, tzinfo=TZ), datetime(2026, 7, 24, 13, 45, tzinfo=TZ)
    )
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.UNAVAILABLE
    assert "city ids" in (quote.note or "")


@respx.mock
async def test_prebound_city_id_skips_autocomplete(flix_search):
    route = respx.get(url__startswith=f"{DEFAULT_BASE}/search/autocomplete/cities").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__startswith=f"{DEFAULT_BASE}/search/service/v4/search").mock(
        return_value=httpx.Response(200, json=flix_search)
    )
    adapter = FlixBusAdapter(min_interval=0.0)
    leg = Leg(
        from_stop=_stop("STO", "Stockholm", 59.33, 18.06, flixbus_city="city-a"),
        to_stop=_stop("GBG", "Goteborg", 57.71, 11.97, flixbus_city="city-b"),
        mode=TransportMode.BUS,
        operator="FlixBus",
        departure=datetime(2026, 7, 24, 7, 30, tzinfo=TZ),
        arrival=datetime(2026, 7, 24, 13, 45, tzinfo=TZ),
    )
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.EXACT
    assert not route.called


async def test_non_flixbus_leg_is_not_supported():
    adapter = FlixBusAdapter()
    leg = _flix_leg(
        datetime(2026, 7, 24, 7, 30, tzinfo=TZ),
        datetime(2026, 7, 24, 13, 45, tzinfo=TZ),
        operator="Vy Bus4You",
    )
    assert not adapter.supports(leg)
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.UNAVAILABLE


# --- Freerider adapter -----------------------------------------------------


@pytest.fixture
def freerider_offers():
    raw = json.loads((FIXTURES / "freerider_sweden.json").read_text(encoding="utf-8"))
    return parse_offers(raw)


def _freerider_leg(offer, service_ref: str | None = None) -> Leg:
    return Leg(
        from_stop=offer.pickup.to_stop(),
        to_stop=offer.dropoff.to_stop(),
        mode=TransportMode.FREERIDER,
        operator="hertz-freerider",
        departure=offer.available_at,
        arrival=offer.available_at + timedelta(seconds=offer.drive_seconds),
        service_ref=service_ref,
    )


async def test_freerider_quote_is_estimated_never_exact(freerider_offers):
    """The tank range and excess-mileage rate are our assumptions, not Hertz's tariff."""
    offer = next(o for o in freerider_offers if o.pickup.city == "Uppsala")
    adapter = FreeriderAdapter()
    adapter.load(freerider_offers)

    quote = await adapter.quote_leg(_freerider_leg(offer, f"freerider:{offer.route_id}@100"))
    assert quote.confidence is PriceConfidence.ESTIMATED
    assert quote.confidence is not PriceConfidence.EXACT
    assert quote.amount_ore == 0
    assert quote.deeplink


async def test_freerider_long_offer_is_not_free(freerider_offers):
    offer = max(freerider_offers, key=lambda o: o.direct_km)
    adapter = FreeriderAdapter()
    adapter.load(freerider_offers)
    quote = await adapter.quote_leg(_freerider_leg(offer))
    assert quote.amount_ore > 0
    assert "assumption" in (quote.note or "").lower()


async def test_freerider_matches_offer_by_service_ref(freerider_offers):
    offer = freerider_offers[0]
    adapter = FreeriderAdapter()
    adapter.load(freerider_offers)
    quote = await adapter.quote_leg(
        _freerider_leg(offer, f"freerider:{offer.route_id}@42000")
    )
    assert quote.confidence is PriceConfidence.ESTIMATED


async def test_freerider_delisted_offer_is_unavailable(freerider_offers):
    offer = freerider_offers[0]
    adapter = FreeriderAdapter()
    adapter.load([])  # inventory refreshed; this car is gone
    quote = await adapter.quote_leg(_freerider_leg(offer, f"freerider:{offer.route_id}@1"))
    assert quote.confidence is PriceConfidence.UNAVAILABLE
    assert "no longer listed" in (quote.note or "")


async def test_freerider_warnings_cover_booking_and_deadline(freerider_offers):
    offer = next(o for o in freerider_offers if o.pickup.city == "Uppsala")
    adapter = FreeriderAdapter()
    adapter.load(freerider_offers)
    warnings = adapter.warnings_for(_freerider_leg(offer, f"freerider:{offer.route_id}@1"))
    blob = " ".join(warnings).lower()
    assert "cannot be reserved from here" in blob
    assert "return by" in blob
    assert "included mileage" in blob


async def test_freerider_adapter_makes_no_http_calls(freerider_offers):
    """Inventory is already in memory; re-fetching per leg would hammer the endpoint."""
    with respx.mock(assert_all_called=False) as mock:
        adapter = FreeriderAdapter()
        adapter.load(freerider_offers)
        await adapter.quote_leg(_freerider_leg(freerider_offers[0]))
        assert not mock.calls


# --- budget ----------------------------------------------------------------


def test_call_budget_stops_runaway_fanout():
    budget = CallBudget(limit=2)
    budget.consume("flixbus")
    budget.consume("flixbus")
    assert budget.remaining("flixbus") == 0
    with pytest.raises(BudgetExceeded):
        budget.consume("flixbus")
    # A different source has its own allowance.
    budget.consume("sj")
    assert budget.spent() == 3


# --- static fares (Flygbussarna airport coaches) ---------------------------


def _fare_leg(operator, frm, to):
    dep = datetime(2026, 7, 22, 8, 0, tzinfo=TZ)
    return Leg(
        from_stop=Stop(id=frm, name=frm, lat=59.0, lon=17.0),
        to_stop=Stop(id=to, name=to, lat=59.0, lon=17.0),
        mode=TransportMode.BUS, operator=operator, departure=dep,
        arrival=dep.replace(hour=9), service_ref="t",
    )


async def test_packaged_flygbussarna_fare_prices_a_coach_leg():
    from tripps.pricing.operators import StaticFareAdapter

    adapter = StaticFareAdapter.load()  # packaged fares only
    assert len(adapter._rows) >= 12
    # Göteborg Nils Ericson -> Landvetter, the real router stop ids.
    leg = _fare_leg("Vy flygbussarna", "740020483", "740000554")
    assert adapter.supports(leg)
    quote = await adapter.quote_leg(leg)
    assert quote.amount_ore == 12900
    assert quote.confidence is PriceConfidence.ESTIMATED
    assert "flygbussarna.se" in (quote.deeplink or "")


async def test_static_fare_ignores_wrong_operator_and_unknown_od():
    from tripps.pricing.operators import StaticFareAdapter

    adapter = StaticFareAdapter.load()
    # right O/D, wrong operator (a FlixBus leg must not borrow a Flygbussarna fare).
    assert not adapter.supports(_fare_leg("Flixbus", "740020483", "740000554"))
    # right operator, unlisted O/D falls through to link-out (unavailable + booking url).
    unknown = _fare_leg("Vy flygbussarna", "740020483", "740000005")
    assert not adapter.supports(unknown)
    quote = await adapter.quote_leg(unknown)
    assert quote.amount_ore is None and quote.deeplink
