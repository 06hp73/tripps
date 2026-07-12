"""The Arlanda C station passage fee: added on top of a leg's fare, even a pass-covered one."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tripps.models import Leg, PriceConfidence, Quote, Stop, TransportMode
from tripps.surcharges import (
    ARLANDA_C_PASSAGE_FEE_ORE,
    ARLANDA_C_STOP_ID,
    apply_arlanda_fee,
)

TZ = ZoneInfo("Europe/Stockholm")

ARLANDA = Stop(id=ARLANDA_C_STOP_ID, name="Arlanda Centralstation", lat=59.65, lon=17.93)
UPPSALA = Stop(id="740000005", name="Uppsala C", lat=59.86, lon=17.65)
STHLM = Stop(id="740000001", name="Stockholm C", lat=59.33, lon=18.06)


def _leg(operator, frm, to, amount=None, source="tora"):
    dep = datetime(2026, 7, 22, 8, 0, tzinfo=TZ)
    leg = Leg(
        from_stop=frm, to_stop=to, mode=TransportMode.TRAIN, operator=operator,
        departure=dep, arrival=dep.replace(hour=9), service_ref="t",
    )
    if amount is not None:
        leg = leg.model_copy(
            update={"quote": Quote(source=source, amount_ore=amount, confidence=PriceConfidence.EXACT)}
        )
    return leg


def test_fee_added_to_regional_leg_boarding_at_arlanda():
    legs, warning = apply_arlanda_fee([_leg("Mälartåg", ARLANDA, UPPSALA, 8000)])
    q = legs[0].quote
    # Kept in surcharge_ore, NOT folded into amount_ore, so coupon identity survives.
    assert q.amount_ore == 8000
    assert q.surcharge_ore == ARLANDA_C_PASSAGE_FEE_ORE
    assert q.total_ore == 8000 + ARLANDA_C_PASSAGE_FEE_ORE
    assert warning is not None and "Arlanda" in warning
    assert q.note is None  # the fee never touches the note (avoids a bogus warning)


def test_fee_added_to_a_pass_covered_leg():
    # A period card zeroes the ride, but the passage fee is still owed.
    covered = _leg("SL", ARLANDA, STHLM, 0, source="travelcard")
    legs, warning = apply_arlanda_fee([covered])
    assert legs[0].quote.amount_ore == 0
    assert legs[0].quote.surcharge_ore == ARLANDA_C_PASSAGE_FEE_ORE
    assert legs[0].quote.total_ore == ARLANDA_C_PASSAGE_FEE_ORE
    assert warning is not None


def test_sj_leg_is_exempt():
    legs, warning = apply_arlanda_fee([_leg("SJ", ARLANDA, UPPSALA, 30000)])
    assert legs[0].quote.amount_ore == 30000  # SJ bundles the passage
    assert legs[0].quote.surcharge_ore is None
    assert warning is None


def test_leg_not_touching_arlanda_is_untouched():
    legs, warning = apply_arlanda_fee([_leg("Mälartåg", UPPSALA, STHLM, 8000)])
    assert legs[0].quote.surcharge_ore is None
    assert warning is None


def test_fee_charged_once_per_itinerary():
    # Two legs both touch Arlanda (arrive, then depart); the barrier is passed once.
    legs, warning = apply_arlanda_fee([
        _leg("Mälartåg", UPPSALA, ARLANDA, 8000),
        _leg("SL", ARLANDA, STHLM, 5000),
    ])
    added = sum((leg.quote.surcharge_ore or 0) for leg in legs)
    assert added == ARLANDA_C_PASSAGE_FEE_ORE
    assert warning is not None


def test_unpriced_leg_is_skipped():
    legs, warning = apply_arlanda_fee([_leg("SL", ARLANDA, STHLM)])  # no quote
    assert legs[0].quote is None
    assert warning is None


def test_itinerary_total_includes_the_surcharge():
    from tripps.models import Itinerary

    legs, _ = apply_arlanda_fee([_leg("Mälartåg", ARLANDA, UPPSALA, 8000)])
    itin = Itinerary(legs=legs)
    assert itin.total_price_ore == 8000 + ARLANDA_C_PASSAGE_FEE_ORE
