"""Tickital second-hand rentals, modelled as an itinerary-level coupon.

A rental is a *paid* second-hand ticket, not an owned pass, so - unlike a travel card - it does
not zero a leg. `TickitalAdapter` gives a covered leg the rental's period price only when no
paid source could, and the orchestrator's coupon then charges that period price once per
itinerary, but only when it beats buying singles (or a covered leg has no purchasable single).
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from tripps.db import Database
from tripps.models import (
    Leg,
    PriceConfidence,
    Quote,
    SearchResponse,
    Stop,
    TransportMode,
)
from tripps.passes import (
    TICKITAL_FARE_CLASS,
    TICKITAL_SOURCE,
    TickitalAdapter,
    TickitalRental,
    tickital_note,
)
from tripps.pricing.orchestrator import PASS_SOURCE, PricingOrchestrator
from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip
from tripps.search import RoundTrip

TZ = ZoneInfo("Europe/Stockholm")

MALMO = Stop(id="MMX", name="Malmö C", lat=55.61, lon=13.00)
LUND = Stop(id="LUND", name="Lund C", lat=55.70, lon=13.19)
HASSLEHOLM = Stop(id="HLM", name="Hässleholm", lat=56.16, lon=13.77)
GOTEBORG = Stop(id="GBG", name="Göteborg C", lat=57.71, lon=11.97)

IN_WINDOW = date(2026, 7, 22)


def _timetable():
    b = TimetableBuilder()
    for s in (MALMO, LUND, HASSLEHOLM, GOTEBORG):
        b.add_stop(s)
    sk = RouteInfo(id="pagatag", mode=TransportMode.TRAIN, operator="Skånetrafiken")
    b.add_trip(sk, ["MMX", "LUND", "HLM"], Trip(id="p1", arrivals=[0, 600, 1200], departures=[0, 600, 1200]))
    ot = RouteInfo(id="ore", mode=TransportMode.TRAIN, operator="Öresundståg")
    b.add_trip(ot, ["MMX", "LUND", "GBG"], Trip(id="o1", arrivals=[0, 600, 9000], departures=[0, 600, 9000]))
    return b.build()


def _leg(operator, frm, to, day=IN_WINDOW, mode=TransportMode.TRAIN):
    dep = datetime(day.year, day.month, day.day, 8, 0, tzinfo=TZ)
    return Leg(
        from_stop=frm, to_stop=to, mode=mode, operator=operator,
        departure=dep, arrival=dep.replace(hour=10), service_ref="t",
    )


def _priced(operator, frm, to, amount, *, source="tora", conf=PriceConfidence.EXACT, day=IN_WINDOW):
    leg = _leg(operator, frm, to, day)
    return leg.model_copy(update={"quote": Quote(source=source, amount_ore=amount, confidence=conf)})


def _rental(**kw):
    base = dict(
        id=1, provider_id="skanetrafiken", price_ore=18000,
        valid_from=date(2026, 7, 20), valid_to=date(2026, 7, 26),
    )
    base.update(kw)
    return TickitalRental(**base)


def _adapter(rental=None):
    a = TickitalAdapter()
    a.prepare(_timetable(), [rental if rental is not None else _rental()])
    return a


def _orch(rental=None):
    return PricingOrchestrator(adapters=[_adapter(rental)])


# --- TickitalAdapter: windowed coverage + fallback quote --------------------


async def test_adapter_supports_covered_leg_in_window():
    a = _adapter()
    assert a.supports(_leg("Öresundståg", MALMO, LUND))


async def test_adapter_does_not_support_out_of_window():
    a = _adapter()
    assert not a.supports(_leg("Öresundståg", MALMO, LUND, day=date(2026, 7, 19)))
    assert not a.supports(_leg("Öresundståg", MALMO, LUND, day=date(2026, 7, 27)))


async def test_adapter_obeys_the_region_gate():
    a = _adapter()
    assert not a.supports(_leg("Öresundståg", MALMO, GOTEBORG))  # leaves Skåne


async def test_adapter_fallback_quote_is_the_period_price():
    a = _adapter()
    quote = await a.quote_leg(_leg("Öresundståg", MALMO, LUND))
    assert quote.source == TICKITAL_SOURCE
    assert quote.amount_ore == 18000
    assert quote.fare_class == TICKITAL_FARE_CLASS
    assert quote.coupon_rental_id == 1
    assert "blocked" in (quote.note or "").lower()


def test_adapter_has_rentals():
    assert _adapter().has_rentals()
    empty = TickitalAdapter()
    empty.prepare(_timetable(), [])
    assert not empty.has_rentals()


def test_tickital_note_states_price_window_and_block_risk():
    from tripps.passes import load_cards

    note = tickital_note(load_cards()["skanetrafiken"], _rental())
    assert "180 SEK" in note
    assert "Jul 20-Jul 26" in note
    assert "blocked" in note.lower()


# --- the coupon: apply only when it beats the singles ----------------------


def test_coupon_not_applied_when_a_single_is_cheaper():
    orch = _orch()
    legs = [_priced("Öresundståg", MALMO, LUND, 4800)]  # 48 SEK single < 180 SEK rental
    out = orch._apply_rental_coupon(legs)
    assert out[0].quote.amount_ore == 4800
    assert out[0].quote.source == "tora"  # untouched


def test_coupon_applied_once_when_it_beats_the_sum_of_singles():
    orch = _orch()
    legs = [
        _priced("Öresundståg", MALMO, LUND, 12000),
        _priced("Skånetrafiken", LUND, HASSLEHOLM, 12000),
    ]  # singles sum 240 SEK > 180 SEK rental
    out = orch._apply_rental_coupon(legs)
    total = sum(leg.quote.amount_ore for leg in out)
    assert total == 18000  # exactly one rental price, not 2x
    assert {leg.quote.source for leg in out} == {TICKITAL_SOURCE}
    charged = [leg for leg in out if leg.quote.amount_ore == 18000]
    assert len(charged) == 1
    assert charged[0].quote.note and "Covered by a tickital" in charged[0].quote.note
    assert all(leg.quote.coupon_rental_id == 1 for leg in out)
    # only the charged leg carries a note, so the warning fires once
    assert sum(1 for leg in out if leg.quote.note) == 1


def test_coupon_tie_keeps_singles():
    orch = _orch()
    legs = [_priced("Öresundståg", MALMO, LUND, 18000)]  # single == rental price
    out = orch._apply_rental_coupon(legs)
    assert out[0].quote.source == "tora"  # tie: no ToS exposure for zero gain


def test_coupon_forced_when_a_covered_leg_has_no_purchasable_single():
    orch = _orch()
    # An unpriceable covered leg (paid adapter returned UNAVAILABLE): the rental is the only
    # complete price, so it applies regardless of the price comparison.
    unavailable = _leg("Skånetrafiken", MALMO, LUND).model_copy(
        update={"quote": Quote.unavailable("tora")}
    )
    out = orch._apply_rental_coupon([unavailable])
    assert out[0].quote.amount_ore == 18000
    assert out[0].quote.source == TICKITAL_SOURCE


async def test_coupon_forced_folds_in_a_cheaper_single_in_the_same_group():
    orch = _orch()
    fallback_quote = await _adapter().quote_leg(_leg("Skånetrafiken", MALMO, LUND))
    fallback_leg = _leg("Skånetrafiken", MALMO, LUND).model_copy(update={"quote": fallback_quote})
    cheap_single = _priced("Öresundståg", LUND, HASSLEHOLM, 3000)  # cheaper than 180 alone
    out = orch._apply_rental_coupon([fallback_leg, cheap_single])
    # fallback in the group forces apply; both covered legs collapse to one 180 SEK charge
    assert sum(leg.quote.amount_ore for leg in out) == 18000


def test_coupon_not_applied_on_a_soft_single():
    orch = _orch(_rental(price_ore=3000))  # rental cheaper than the single...
    legs = [_priced("Öresundståg", MALMO, LUND, 4800, conf=PriceConfidence.STALE)]
    out = orch._apply_rental_coupon(legs)
    # ...but the single is STALE (not firm), so we do not push into a terms-risky rental
    assert out[0].quote.source == "tora"
    assert out[0].quote.confidence is PriceConfidence.STALE


def test_coupon_skips_a_leg_an_owned_card_already_freed():
    orch = _orch()
    owned = _leg("Öresundståg", MALMO, LUND).model_copy(
        update={"quote": Quote(source=PASS_SOURCE, amount_ore=0, confidence=PriceConfidence.EXACT)}
    )
    out = orch._apply_rental_coupon([owned])
    assert out[0].quote.source == PASS_SOURCE  # untouched, never folded into a rental
    assert out[0].quote.amount_ore == 0


def test_coupon_no_op_without_rentals():
    orch = PricingOrchestrator(adapters=[TickitalAdapter()])
    legs = [_priced("Öresundståg", MALMO, LUND, 12000)]
    out = orch._apply_rental_coupon(legs)
    assert out is legs  # nothing bound, returns the same list


def test_zone_combination_hint_fires_for_a_partially_covered_leg():
    from tripps.passes import PassAdapter

    pa = PassAdapter()
    pa.prepare(_timetable(), ["skanetrafiken"])
    orch = PricingOrchestrator(adapters=[pa])
    # Malmö (Skåne) -> Göteborg (out of Skåne) on Öresundståg, priced by a paid source.
    partial = _priced("Öresundståg", MALMO, GOTEBORG, 23700, source="tora")
    warnings = orch._warnings_for([partial])
    assert any("Skånetrafiken card may reduce" in w for w in warnings)


def test_no_zone_hint_for_a_fully_covered_or_uncovered_leg():
    from tripps.passes import PassAdapter

    pa = PassAdapter()
    pa.prepare(_timetable(), ["skanetrafiken"])
    orch = PricingOrchestrator(adapters=[pa])
    # Fully in-region (would be free anyway): no hint.
    full = _priced("Öresundståg", MALMO, LUND, 4800, source="tora")
    assert not any("may reduce" in w for w in orch._warnings_for([full]))
    # A leg already zeroed by the card (source travelcard) must not get the hint either.
    covered = _leg("Öresundståg", MALMO, GOTEBORG).model_copy(
        update={"quote": Quote(source=PASS_SOURCE, amount_ore=0, confidence=PriceConfidence.EXACT)}
    )
    assert not any("may reduce" in w for w in orch._warnings_for([covered]))


def test_no_floor_violation_for_a_zeroed_rental_operator():
    # A rental-covered leg now priced by a paid source (Tora) below the calibrated floor: the
    # router used a 0 floor for that operator, so it pruned nothing and this is not a violation.
    from tripps.pricing.base import CallBudget
    from tripps.pricing.orchestrator import _Context
    from tripps.routing.journey import leg_distance_km

    orch = PricingOrchestrator(adapters=[])
    leg = _priced("Öresundståg", MALMO, GOTEBORG, 2000, source="tora")  # 20 SEK, below floor
    assert orch.floors.floor_ore(leg.mode, leg.operator, leg_distance_km(leg)) > 2000

    def ctx(zeroed):
        return _Context(call_budget=CallBudget.from_settings(orch.budget), zeroed_operators=zeroed)

    not_zeroed = ctx(frozenset())
    orch._check_floor(leg, leg.quote, not_zeroed)
    assert not_zeroed.violations  # would be flagged without the fix

    zeroed = ctx(frozenset({"Öresundståg"}))
    orch._check_floor(leg, leg.quote, zeroed)
    assert not zeroed.violations  # skipped: the operator's floor was zeroed for this search


# --- round-trip dedup: a one-time period cost is counted once --------------


def _charged_itin(amount, rental_id):
    leg = _leg("Öresundståg", MALMO, LUND).model_copy(
        update={
            "quote": Quote(
                source=TICKITAL_SOURCE, amount_ore=amount,
                confidence=PriceConfidence.EXACT, fare_class=TICKITAL_FARE_CLASS,
                coupon_rental_id=rental_id,
            )
        }
    )
    from tripps.models import Itinerary

    return Itinerary(legs=[leg])


def _response(itin):
    return SearchResponse(
        origin=MALMO, destination=LUND, date=IN_WINDOW.isoformat(),
        itineraries=[itin], warnings=[], source_status={},
    )


def test_round_trip_counts_a_shared_rental_once():
    rental = _rental(price_ore=18000)
    rt = RoundTrip(
        outbound=_response(_charged_itin(18000, 1)),
        inbound=_response(_charged_itin(18000, 1)),
        tickital_rentals=[rental],
    )
    # naive sum would be 360 SEK; the one-time period cost is counted once => 180 SEK
    assert rt.total_price_ore == 18000
    assert rt.shared_rentals == {1}
    assert rt.warnings and "once" in rt.warnings[0]


def test_round_trip_does_not_dedup_when_only_one_direction_uses_it():
    rt = RoundTrip(
        outbound=_response(_charged_itin(18000, 1)),
        inbound=_response(_charged_itin(5000, 2)),  # a different rental
        tickital_rentals=[_rental(id=1, price_ore=18000), _rental(id=2, price_ore=5000)],
    )
    assert rt.shared_rentals == set()
    assert rt.total_price_ore == 18000 + 5000


def test_round_trip_dedup_survives_an_arlanda_surcharge_on_the_charged_leg():
    # The Arlanda fee lands in surcharge_ore, not amount_ore, so amount_ore == price_ore holds
    # and the charged rental is still detected in both directions.
    def charged_with_fee():
        leg = _leg("Öresundståg", MALMO, LUND).model_copy(
            update={
                "quote": Quote(
                    source=TICKITAL_SOURCE, amount_ore=18000, surcharge_ore=15700,
                    confidence=PriceConfidence.EXACT, fare_class=TICKITAL_FARE_CLASS,
                    coupon_rental_id=1,
                )
            }
        )
        from tripps.models import Itinerary

        return Itinerary(legs=[leg])

    rt = RoundTrip(
        outbound=_response(charged_with_fee()),
        inbound=_response(charged_with_fee()),
        tickital_rentals=[_rental(id=1, price_ore=18000)],
    )
    assert rt.shared_rentals == {1}  # surcharge did not hide the charge
    # each direction total = 18000 + 15700; the one-time period cost is deducted once
    assert rt.total_price_ore == (18000 + 15700) * 2 - 18000


def test_no_spurious_tickital_warning_when_fee_lands_on_a_zeroed_sibling():
    orch = _orch()
    from tripps.passes import load_cards

    note = tickital_note(load_cards()["skanetrafiken"], _rental())
    charged = _leg("Skånetrafiken", MALMO, LUND).model_copy(
        update={"quote": Quote(source=TICKITAL_SOURCE, amount_ore=18000, note=note,
                               confidence=PriceConfidence.EXACT,
                               fare_class=TICKITAL_FARE_CLASS, coupon_rental_id=1)}
    )
    # a zeroed sibling that later received an Arlanda surcharge but no note
    sibling = _leg("Skånetrafiken", LUND, HASSLEHOLM).model_copy(
        update={"quote": Quote(source=TICKITAL_SOURCE, amount_ore=0, surcharge_ore=15700,
                               note=None, confidence=PriceConfidence.EXACT,
                               fare_class=TICKITAL_FARE_CLASS, coupon_rental_id=1)}
    )
    warnings = orch._warnings_for([charged, sibling])
    tickital_warnings = [w for w in warnings if "tickital" in w.lower()]
    assert len(tickital_warnings) == 1  # exactly once, from the charged leg's note


# --- DB CRUD ---------------------------------------------------------------


def test_db_add_list_active_remove(tmp_path):
    db = Database(tmp_path / "t.db")
    try:
        rid = db.add_tickital_rental(
            provider_id="skanetrafiken", price_ore=18000,
            valid_from=date(2026, 7, 20), valid_to=date(2026, 7, 26), note="ad 42",
        )
        assert rid > 0
        rows = db.list_tickital_rentals()
        assert len(rows) == 1
        rental = TickitalRental.from_row(rows[0])
        assert rental.provider_id == "skanetrafiken" and rental.price_ore == 18000
        assert rental.valid_from == date(2026, 7, 20) and rental.note == "ad 42"
        assert len(db.active_tickital_rentals(date(2026, 7, 22))) == 1
        assert db.active_tickital_rentals(date(2026, 7, 27)) == []
        assert db.remove_tickital_rental(rid) is True
        assert db.list_tickital_rentals() == []
        assert db.remove_tickital_rental(rid) is False
    finally:
        db.close()
