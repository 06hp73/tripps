"""Travel-card coverage: the region gate that stops a consortium train being wrongly freed."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tripps.models import Leg, PriceConfidence, Stop, TransportMode
from tripps.passes import (
    GLOBAL_EXCLUSIONS,
    PassAdapter,
    PassCoverage,
    load_cards,
)
from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip

TZ = ZoneInfo("Europe/Stockholm")

# Skåne stations + one Västra Götaland station, so a region gate has something to reject.
MALMO = Stop(id="MMX", name="Malmö C", lat=55.61, lon=13.00)
LUND = Stop(id="LUND", name="Lund C", lat=55.70, lon=13.19)
HASSLEHOLM = Stop(id="HLM", name="Hässleholm", lat=56.16, lon=13.77)
GOTEBORG = Stop(id="GBG", name="Göteborg C", lat=57.71, lon=11.97)


def _timetable():
    b = TimetableBuilder()
    for s in (MALMO, LUND, HASSLEHOLM, GOTEBORG):
        b.add_stop(s)
    # Skånetrafiken (Pågatåg) within Skåne — defines the Skåne region-stop set.
    sk = RouteInfo(id="pagatag", mode=TransportMode.TRAIN, operator="Skånetrafiken")
    b.add_trip(sk, ["MMX", "LUND", "HLM"], Trip(id="p1", arrivals=[0, 600, 1200], departures=[0, 600, 1200]))
    # Öresundståg runs Malmö all the way to Göteborg (out of Skåne).
    ot = RouteInfo(id="ore", mode=TransportMode.TRAIN, operator="Öresundståg")
    b.add_trip(ot, ["MMX", "LUND", "GBG"], Trip(id="o1", arrivals=[0, 600, 9000], departures=[0, 600, 9000]))
    return b.build()


def _leg(operator, frm, to, mode=TransportMode.TRAIN):
    dep = datetime(2026, 7, 22, 8, 0, tzinfo=TZ)
    return Leg(
        from_stop=frm, to_stop=to, mode=mode, operator=operator,
        departure=dep, arrival=dep.replace(hour=10), service_ref="t",
    )


@pytest.fixture
def coverage():
    return PassCoverage(_timetable())


# --- registry --------------------------------------------------------------


def test_registry_loads_the_major_providers():
    cards = load_cards()
    for cid in ("skanetrafiken", "vasttrafik", "sl-stockholm", "ostgotatrafiken"):
        assert cid in cards, cid
    assert "SJ" not in cards["skanetrafiken"].honored_operators, "SJ is never auto-freed"


def test_no_card_honors_a_globally_excluded_operator():
    for card in load_cards().values():
        assert not (card.honored_operators & GLOBAL_EXCLUSIONS), card.id


# --- the region gate (the load-bearing test) -------------------------------


def test_skanetrafiken_frees_oresundstag_within_skane(coverage):
    # Malmö -> Lund is entirely inside Skåne.
    assert coverage.covers("skanetrafiken", _leg("Öresundståg", MALMO, LUND))


def test_skanetrafiken_does_not_free_oresundstag_leaving_skane(coverage):
    # Malmö -> Göteborg leaves Skåne; a Skånetrafiken card must NOT zero it.
    assert not coverage.covers("skanetrafiken", _leg("Öresundståg", MALMO, GOTEBORG))


def test_home_operator_is_covered_within_region(coverage):
    assert coverage.covers("skanetrafiken", _leg("Skånetrafiken", MALMO, HASSLEHOLM))


def test_operator_not_honored_is_not_covered(coverage):
    # Västtrafik is not in the Skånetrafiken honored set.
    assert not coverage.covers("skanetrafiken", _leg("Västtrafik", MALMO, LUND))


def test_globally_excluded_operator_never_covered(coverage):
    # Even if some card's data listed SJ, the global gate blocks it.
    leg = _leg("SJ", MALMO, LUND)
    assert not coverage.covers("skanetrafiken", leg)
    assert not coverage.covers("vasttrafik", leg)


def test_walk_leg_is_never_covered(coverage):
    assert not coverage.covers("skanetrafiken", _leg("Skånetrafiken", MALMO, LUND, mode=TransportMode.WALK))


# --- operator-only model ---------------------------------------------------


def test_operator_only_card_needs_no_region_gate():
    b = TimetableBuilder()
    a = Stop(id="A", name="Stockholm City", lat=59.33, lon=18.06)
    z = Stop(id="Z", name="Bålsta", lat=59.57, lon=17.53)
    b.add_stop(a)
    b.add_stop(z)
    b.add_trip(
        RouteInfo(id="pt", mode=TransportMode.TRAIN, operator="SL"),
        ["A", "Z"], Trip(id="t", arrivals=[0, 600], departures=[0, 600]),
    )
    cov = PassCoverage(b.build())
    assert cov.covers("sl-stockholm", _leg("SL", a, z))
    assert cov.is_supported("sl-stockholm")


# --- support / limited -----------------------------------------------------


def test_card_with_no_region_in_feed_is_unsupported(coverage):
    # Hallandstrafiken has no home agency in this mini feed, so it cannot free anything.
    assert not coverage.is_supported("hallandstrafiken")
    assert not coverage.covers("hallandstrafiken", _leg("Öresundståg", MALMO, LUND))


def test_supported_flag_true_for_a_present_region(coverage):
    assert coverage.is_supported("skanetrafiken")


# --- covering_card ---------------------------------------------------------


def test_covering_card_returns_the_matching_card(coverage):
    card = coverage.covering_card(["vasttrafik", "skanetrafiken"], _leg("Öresundståg", MALMO, LUND))
    assert card is not None and card.id == "skanetrafiken"


def test_covering_card_none_when_no_card_covers(coverage):
    assert coverage.covering_card(["vasttrafik"], _leg("Öresundståg", MALMO, LUND)) is None


# --- PassAdapter -----------------------------------------------------------


async def test_pass_adapter_prices_covered_leg_free():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), ["skanetrafiken"])
    leg = _leg("Öresundståg", MALMO, LUND)
    assert adapter.supports(leg)
    quote = await adapter.quote_leg(leg)
    assert quote.confidence is PriceConfidence.EXACT
    assert quote.amount_ore == 0
    assert "Skånetrafiken" in (quote.note or "")


async def test_pass_adapter_does_not_support_uncovered_leg():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), ["skanetrafiken"])
    # Leaving Skåne — falls through to paid adapters.
    assert not adapter.supports(_leg("Öresundståg", MALMO, GOTEBORG))


async def test_pass_adapter_with_no_cards_supports_nothing():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), [])
    assert not adapter.supports(_leg("Öresundståg", MALMO, LUND))


def test_pass_adapter_provides_price():
    assert PassAdapter().provides_price is True
