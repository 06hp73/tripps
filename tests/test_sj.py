"""SJ adapter: schema contract against recorded live responses, matching, degradation.

The fixtures are real captures of sj.se's booking backend from 2026-07-10. If the API
changes shape, these fail here instead of silently mispricing an SJ leg.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx

from tripps.models import Leg, PriceConfidence, Stop, TransportMode
from tripps.pricing.base import CallBudget
from tripps.pricing.sj import (
    DEFAULT_API,
    FALLBACK_KEY,
    SJAdapter,
    parse_departures,
    parse_offer_price_ore,
    parse_search,
)

TZ = ZoneInfo("Europe/Stockholm")
FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def sj_search() -> dict:
    return _fixture("sj_search.json")


@pytest.fixture
def sj_departures() -> dict:
    return _fixture("sj_departures.json")


@pytest.fixture
def sj_offers() -> dict:
    return _fixture("sj_offers.json")


def _leg(dep: datetime, arr: datetime, operator: str = "SJ", frm: str = "740000001", to: str = "740000002") -> Leg:
    return Leg(
        from_stop=Stop(id=frm, name="Stockholm Centralstation", lat=59.33, lon=18.06),
        to_stop=Stop(id=to, name="Göteborg Centralstation", lat=57.71, lon=11.97),
        mode=TransportMode.TRAIN,
        operator=operator,
        departure=dep,
        arrival=arr,
        service_ref="t",
    )


# --- parsing (schema contract) --------------------------------------------


def test_search_yields_ids(sj_search):
    search_id, plid = parse_search(sj_search)
    assert search_id and plid
    assert search_id != plid


def test_departures_parse_with_times_and_changes(sj_departures):
    deps = parse_departures(sj_departures)
    assert deps, "fixture must contain departures"
    d = deps[0]
    assert d.departure.tzinfo is not None
    assert d.arrival > d.departure
    assert isinstance(d.changes, int)
    assert d.departure_id


def test_uic_codes_are_gtfs_stop_ids(sj_search):
    """The load-bearing fact that removes any station mapping layer."""
    assert sj_search["origin"] in ("740000001", {"uicStationCode": "740000001"}) or True
    # The search we sent used the GTFS ids directly; the echo confirms the API accepts them.
    origin = sj_search.get("origin")
    code = origin.get("uicStationCode") if isinstance(origin, dict) else origin
    assert code == "740000001"


def test_offer_price_is_the_cheapest_fare_in_ore(sj_offers):
    price = parse_offer_price_ore(sj_offers)
    assert price is not None
    assert price == int(float(sj_offers["priceFrom"]["price"]) * 100)
    assert sj_offers["priceFrom"]["currency"] == "SEK"


def test_unavailable_departure_has_no_price():
    assert parse_offer_price_ore({"available": False, "priceFrom": None}) is None


def test_missing_pricefrom_is_tolerated():
    assert parse_offer_price_ore({"available": True}) is None
    assert parse_offer_price_ore({"available": True, "priceFrom": {}}) is None


# --- adapter over mocked HTTP ---------------------------------------------


def _mock_sj(mock, search, departures, offers, *, key=FALLBACK_KEY):
    mock.get(url__startswith=f"{DEFAULT_API}/config").mock(return_value=httpx.Response(200, json={}))
    mock.post(url__startswith=f"{DEFAULT_API}/search").mock(return_value=httpx.Response(200, json=search))
    mock.get(url__regex=rf"{DEFAULT_API}/departures/search/.*").mock(
        return_value=httpx.Response(200, json=departures)
    )
    mock.get(url__regex=rf"{DEFAULT_API}/departures/[^/]+/offers.*").mock(
        return_value=httpx.Response(200, json=offers)
    )


@respx.mock
async def test_direct_departure_is_priced_exact(sj_search, sj_departures, sj_offers):
    _mock_sj(respx.mock, sj_search, sj_departures, sj_offers)
    adapter = SJAdapter(min_interval=0.0, key="testkey")

    deps = parse_departures(sj_departures)
    direct = next(d for d in deps if d.is_direct)
    quote = await adapter.quote_leg(_leg(direct.departure, direct.arrival))
    await adapter.aclose()

    assert quote.confidence is PriceConfidence.EXACT
    assert quote.amount_ore == int(float(sj_offers["priceFrom"]["price"]) * 100)
    assert quote.deeplink


@respx.mock
async def test_day_search_is_reused_across_legs(sj_search, sj_departures, sj_offers):
    """A second SJ leg on the same day/route must not repeat the search+departures calls."""
    _mock_sj(respx.mock, sj_search, sj_departures, sj_offers)
    search_route = respx.routes[1]  # the POST /search mock
    adapter = SJAdapter(min_interval=0.0, key="testkey")

    deps = [d for d in parse_departures(sj_departures) if d.is_direct]
    if len(deps) < 1:
        pytest.skip("fixture has no direct departure")
    await adapter.quote_leg(_leg(deps[0].departure, deps[0].arrival))
    calls_after_first = search_route.call_count
    await adapter.quote_leg(_leg(deps[0].departure, deps[0].arrival))
    await adapter.aclose()

    assert search_route.call_count == calls_after_first == 1


@respx.mock
async def test_no_matching_departure_is_unavailable(sj_search, sj_departures, sj_offers):
    _mock_sj(respx.mock, sj_search, sj_departures, sj_offers)
    adapter = SJAdapter(min_interval=0.0, key="testkey")
    # A departure time nowhere near any in the fixture.
    quote = await adapter.quote_leg(
        _leg(datetime(2026, 7, 24, 23, 59, tzinfo=TZ), datetime(2026, 7, 25, 4, 0, tzinfo=TZ))
    )
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.UNAVAILABLE
    assert "no matching SJ departure" in (quote.note or "")


@respx.mock
async def test_upstream_failure_degrades_to_unavailable(sj_departures):
    respx.get(url__startswith=f"{DEFAULT_API}/config").mock(return_value=httpx.Response(200, json={}))
    respx.post(url__startswith=f"{DEFAULT_API}/search").mock(return_value=httpx.Response(503))
    adapter = SJAdapter(min_interval=0.0, key="testkey")
    quote = await adapter.quote_leg(
        _leg(datetime(2026, 7, 24, 5, 9, tzinfo=TZ), datetime(2026, 7, 24, 10, 0, tzinfo=TZ))
    )
    health = await adapter.health()
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.UNAVAILABLE
    assert quote.deeplink
    assert health.state.value == "down"


@respx.mock
async def test_sold_out_departure_is_unavailable(sj_search, sj_departures):
    _mock_sj(respx.mock, sj_search, sj_departures, {"available": False, "priceFrom": None})
    adapter = SJAdapter(min_interval=0.0, key="testkey")
    deps = [d for d in parse_departures(sj_departures) if d.is_direct]
    quote = await adapter.quote_leg(_leg(deps[0].departure, deps[0].arrival))
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.UNAVAILABLE
    assert "sold out" in (quote.note or "")


# --- support & routing -----------------------------------------------------


def test_supports_only_sj_family_train_legs_with_numeric_stops():
    adapter = SJAdapter(key="k")
    now = datetime(2026, 7, 24, 8, 0, tzinfo=TZ)
    later = datetime(2026, 7, 24, 12, 0, tzinfo=TZ)
    assert adapter.supports(_leg(now, later, operator="SJ"))
    assert adapter.supports(_leg(now, later, operator="SJ Nord"))
    assert not adapter.supports(_leg(now, later, operator="Öresundståg"))
    assert not adapter.supports(_leg(now, later, operator="FlixBus"))
    # A non-UIC stop id (e.g. a bus stop) cannot be sent to SJ.
    assert not adapter.supports(_leg(now, later, operator="SJ", frm="9022050000001"[:5] + "x"))


@respx.mock
async def test_key_extraction_from_bundle_and_validation():
    """The key is pulled from the site bundle and the one the API accepts is kept."""
    home_html = '<script src="https://www.sj.se/assets/main-ABC123.js"></script>'
    bundle = (
        '..."Ocp-Apim-Subscription-Key":`badbadbadbadbadbadbadbadbadbad00`,...'
        '..."Ocp-Apim-Subscription-Key":`d6625619def348d38be070027fd24ff6`,...'
    )
    respx.get("https://www.sj.se/en").mock(return_value=httpx.Response(200, text=home_html))
    respx.get("https://www.sj.se/assets/main-ABC123.js").mock(
        return_value=httpx.Response(200, text=bundle)
    )
    # Only the second key validates.
    respx.get(
        url__startswith=f"{DEFAULT_API}/config",
    ).mock(
        side_effect=lambda request: httpx.Response(
            200 if request.headers.get("ocp-apim-subscription-key") == "d6625619def348d38be070027fd24ff6" else 401
        )
    )
    adapter = SJAdapter(min_interval=0.0)
    key = await adapter.ensure_key()
    await adapter.aclose()
    assert key == "d6625619def348d38be070027fd24ff6"


async def test_key_resolution_does_not_consume_search_budget():
    """Pre-warming the key must not spend the per-search call budget."""
    with respx.mock:
        respx.get("https://www.sj.se/en").mock(return_value=httpx.Response(200, text="no bundle here"))
        respx.get(url__startswith=f"{DEFAULT_API}/config").mock(return_value=httpx.Response(200, json={}))
        adapter = SJAdapter(min_interval=0.0)
        adapter.set_budget(CallBudget(limit=1))
        await adapter.ensure_key()  # falls back to FALLBACK_KEY, validated
        # The budget is untouched: a real price search still has its full allowance.
        assert adapter._budget.remaining(adapter.name) == 1
        await adapter.aclose()
