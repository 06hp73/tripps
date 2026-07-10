"""Tora (Trainplanet) adapter: schema contract against a recorded response, matching, budget.

The fixture is a real capture of `wl.tora.trainplanet.com/v1/offers` from 2026-07-10 for
Malmo->Goteborg, which carries Oresundstag, SJ and Vy Bus4You journeys. HTTP is mocked by
monkeypatching the sync `primp` fetch, since `primp` cannot be intercepted by `respx`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tripps.models import Leg, PriceConfidence, Stop, TransportMode
from tripps.pricing.base import CallBudget
from tripps.pricing.tora import (
    ToraAdapter,
    carrier_matches,
    parse_offers,
    place_urn,
)

TZ = ZoneInfo("Europe/Stockholm")
FIXTURE = Path(__file__).parent / "fixtures" / "tora_offers.json"


@pytest.fixture
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _leg(operator, dep, arr, mode=TransportMode.TRAIN, frm="740000003", to="740000002") -> Leg:
    return Leg(
        from_stop=Stop(id=frm, name="Malmö C", lat=55.6, lon=13.0),
        to_stop=Stop(id=to, name="Göteborg C", lat=57.7, lon=12.0),
        mode=mode,
        operator=operator,
        departure=dep,
        arrival=arr,
        service_ref="t",
    )


def _adapter_returning(payload: dict) -> ToraAdapter:
    """A ToraAdapter whose network call is replaced by the fixture."""
    adapter = ToraAdapter(min_interval=0.0)

    def fake_fetch(origin, destination, when):
        return payload

    adapter._fetch_offers_sync = fake_fetch  # type: ignore[method-assign]
    return adapter


# --- helpers ---------------------------------------------------------------


def test_place_urn_is_built_from_the_gtfs_id():
    assert place_urn("740000003") == "urn:x_swe:stn:740000003"


def test_carrier_matching_is_case_and_spacing_tolerant():
    assert carrier_matches("Vy Bus4You", "Vy bus4you")
    assert carrier_matches("Öresundståg", "Öresundståg")
    assert carrier_matches("Mälartåg", "Mälartåg")
    assert not carrier_matches("SJ", "Öresundståg")
    assert not carrier_matches(None, "SJ")


# --- parsing (schema contract) --------------------------------------------


def test_parses_journeys_with_carriers_and_prices(payload):
    journeys = parse_offers(payload)
    assert len(journeys) == 4
    first = journeys[0]
    assert first.departure.tzinfo is not None
    assert first.arrival > first.departure
    assert first.sole_carrier == "Öresundståg"
    assert first.cheapest_ore == 43_500  # 435 SEK, in öre


def test_multiple_carriers_are_present(payload):
    """The load-bearing fact: one response prices several operators."""
    carriers = {j.sole_carrier for j in parse_offers(payload)}
    assert {"Öresundståg", "SJ", "Vy bus4you"} <= carriers


def test_zero_amount_entries_are_not_treated_as_free(payload):
    for j in parse_offers(payload):
        assert j.cheapest_ore is None or j.cheapest_ore > 0


def test_malformed_journey_is_skipped():
    assert parse_offers({"journeys": [{"trip": {}}, {"nonsense": 1}]}) == []


# --- matching + quoting ----------------------------------------------------


async def test_oresundstag_leg_is_priced_exact(payload):
    adapter = _adapter_returning(payload)
    leg = _leg("Öresundståg", datetime(2026, 7, 24, 8, 12, tzinfo=TZ), datetime(2026, 7, 24, 11, 5, tzinfo=TZ))
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.EXACT
    assert quote.amount_ore == 43_500
    assert "Öresundståg" in (quote.note or "")


async def test_bus_operator_is_priced_from_the_same_response(payload):
    """Vy Bus4You is a bus; Tora prices it alongside the trains."""
    adapter = _adapter_returning(payload)
    leg = _leg(
        "Vy Bus4You",
        datetime(2026, 7, 24, 9, 50, tzinfo=TZ),
        datetime(2026, 7, 24, 13, 15, tzinfo=TZ),
        mode=TransportMode.BUS,
    )
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.EXACT
    assert quote.amount_ore == 22_400


async def test_carrier_mismatch_does_not_borrow_another_operators_price(payload):
    """A leg whose operator is not among the returned carriers must not be priced with
    whatever happened to be cheapest."""
    adapter = _adapter_returning(payload)
    leg = _leg("Krösatåg", datetime(2026, 7, 24, 8, 12, tzinfo=TZ), datetime(2026, 7, 24, 11, 5, tzinfo=TZ))
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.UNAVAILABLE


async def test_departure_far_from_any_journey_is_unavailable(payload):
    adapter = _adapter_returning(payload)
    leg = _leg("Öresundståg", datetime(2026, 7, 24, 23, 0, tzinfo=TZ), datetime(2026, 7, 25, 2, 0, tzinfo=TZ))
    quote = await adapter.quote_leg(leg)
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.UNAVAILABLE


async def test_day_response_is_cached_across_legs(payload):
    calls = {"n": 0}
    adapter = ToraAdapter(min_interval=0.0)

    def fake_fetch(origin, destination, when):
        calls["n"] += 1
        return payload

    adapter._fetch_offers_sync = fake_fetch  # type: ignore[method-assign]

    await adapter.quote_leg(_leg("Öresundståg", datetime(2026, 7, 24, 8, 12, tzinfo=TZ), datetime(2026, 7, 24, 11, 5, tzinfo=TZ)))
    await adapter.quote_leg(_leg("Vy Bus4You", datetime(2026, 7, 24, 9, 50, tzinfo=TZ), datetime(2026, 7, 24, 13, 15, tzinfo=TZ), mode=TransportMode.BUS))
    await adapter.aclose()
    assert calls["n"] == 1, "both legs share one day response"


async def test_upstream_failure_degrades_to_unavailable():
    adapter = ToraAdapter(min_interval=0.0)

    def boom(origin, destination, when):
        raise ValueError("tora offers returned HTTP 403")

    adapter._fetch_offers_sync = boom  # type: ignore[method-assign]
    quote = await adapter.quote_leg(_leg("Öresundståg", datetime(2026, 7, 24, 8, 12, tzinfo=TZ), datetime(2026, 7, 24, 11, 5, tzinfo=TZ)))
    health = await adapter.health()
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.UNAVAILABLE
    assert quote.deeplink
    assert health.state.value == "down"


# --- support gate ----------------------------------------------------------


def test_supports_regional_operators_not_sj_or_flixbus():
    adapter = ToraAdapter()
    dep = datetime(2026, 7, 24, 8, 0, tzinfo=TZ)
    arr = datetime(2026, 7, 24, 10, 0, tzinfo=TZ)
    assert adapter.supports(_leg("Öresundståg", dep, arr))
    assert adapter.supports(_leg("Mälartåg", dep, arr))
    assert adapter.supports(_leg("Vy Bus4You", dep, arr, mode=TransportMode.BUS))
    # SJ and FlixBus have their own, closer-to-source adapters.
    assert not adapter.supports(_leg("SJ", dep, arr))
    assert not adapter.supports(_leg("FlixBus", dep, arr, mode=TransportMode.BUS))


def test_provides_price_is_true():
    """Unlike the link-out adapter, Tora returns real amounts."""
    assert getattr(ToraAdapter(), "provides_price", True) is True


async def test_budget_is_charged_per_day_lookup(payload):
    adapter = _adapter_returning(payload)
    adapter.set_budget(CallBudget(limit=5))
    await adapter.quote_leg(_leg("Öresundståg", datetime(2026, 7, 24, 8, 12, tzinfo=TZ), datetime(2026, 7, 24, 11, 5, tzinfo=TZ)))
    assert adapter._budget.used.get("tora") == 1
    await adapter.aclose()


async def test_budget_exhaustion_does_not_mark_the_source_down(payload):
    """Running out of per-search calls is not the endpoint being down."""
    adapter = _adapter_returning(payload)
    adapter.set_budget(CallBudget(limit=0))  # no calls allowed
    quote = await adapter.quote_leg(
        _leg("Öresundståg", datetime(2026, 7, 24, 8, 12, tzinfo=TZ), datetime(2026, 7, 24, 11, 5, tzinfo=TZ))
    )
    health = await adapter.health()
    await adapter.aclose()
    assert quote.confidence is PriceConfidence.UNAVAILABLE
    assert "exhausted" in (quote.note or "")
    assert health.state.value == "ok", "budget exhaustion must not read as a dead source"
