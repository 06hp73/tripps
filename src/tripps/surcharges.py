"""Station fees that ride on top of a leg's fare, whoever priced it.

The Arlanda C station passage fee is the load-bearing case. The railway station under
Stockholm Arlanda airport (GTFS stop 740000556, "Arlanda Centralstation") sits behind a
ticket barrier owned by A-Train. Boarding or alighting there on a regional/commuter operator
(SL pendeltåg, Mälartåg, UL) costs a passage fee *on top of* the train fare - and a regional
period card does NOT cover it. SJ and Arlanda Express bundle the passage into their own fare,
so their legs owe nothing extra.

So this fee cannot live in a price adapter: it must be added after pricing, to whatever the
leg ended up costing - a paid single, a zeroed owned-card leg, or a tickital-coupon leg alike.
It is added once per itinerary (one barrier passage), to the first non-exempt leg that touches
Arlanda C.

The amount is a documented, configurable constant, not a guess: 157 SEK as published by
Swedavia (swedavia.com/arlanda/trains) for 2026. A-Train raises it most years, so review it
annually rather than trusting it silently.
"""

from __future__ import annotations

from .models import SEK, Leg, TransportMode

#: GTFS stop_id of Arlanda Centralstation (the regional/commuter platforms behind the barrier).
#: The separate Arlanda Express platforms (740000492 / 740000708) are NOT this stop, so an
#: Arlanda Express leg never touches it and is exempt automatically.
ARLANDA_C_STOP_ID = "740000556"

#: Arlanda C station passage fee. Source: swedavia.com/arlanda/trains, valid 2026. Set by
#: A-Train (Arlanda Express owner); it rises most years, so review this yearly.
ARLANDA_C_PASSAGE_FEE_ORE = 157 * SEK

#: Operators whose fare already includes the Arlanda passage, so their legs owe no extra fee.
ARLANDA_FEE_EXEMPT_OPERATORS = frozenset({"SJ", "SJ Nord", "Arlanda Express"})


def _fee_applies(leg: Leg) -> bool:
    if leg.mode is TransportMode.WALK:
        return False
    touches = leg.from_stop.id == ARLANDA_C_STOP_ID or leg.to_stop.id == ARLANDA_C_STOP_ID
    return touches and (leg.operator or "") not in ARLANDA_FEE_EXEMPT_OPERATORS


def apply_arlanda_fee(legs: list[Leg]) -> tuple[list[Leg], str | None]:
    """Add the Arlanda C passage fee once, returning (legs, warning-or-None).

    The fee lands on the first non-exempt leg that boards or alights at Arlanda C and already
    carries a real amount (a pass-covered zero counts - the card does not cover the passage).
    An unpriced leg is skipped: the itinerary is dropped for being unpriced anyway, so there is
    nothing to add onto. Charged once per itinerary (one barrier passage), even if two legs
    touch the station.

    The fee is written to `quote.surcharge_ore`, NOT folded into `quote.amount_ore`, and the
    leg's `note` is left untouched. That keeps two things intact: the coupon's
    `amount_ore == rental.price_ore` identity (so cross-itinerary dedup still finds the charged
    leg), and `_warnings_for`'s note-based tickital warning (so a fee landing on a rental leg
    does not resurface as a bogus terms warning). The fee is surfaced to the user by the
    returned warning and by `surcharge_ore` in the itinerary total.
    """
    for i, leg in enumerate(legs):
        if not _fee_applies(leg):
            continue
        quote = leg.quote
        if quote is None or quote.amount_ore is None:
            continue
        fee_sek = ARLANDA_C_PASSAGE_FEE_ORE // SEK
        updated = list(legs)
        updated[i] = leg.model_copy(
            update={"quote": quote.model_copy(update={"surcharge_ore": ARLANDA_C_PASSAGE_FEE_ORE})}
        )
        warning = (
            f"Arlanda C station passage fee of {fee_sek} SEK added: it is not included in a "
            "regional period ticket (SJ and Arlanda Express fares already include it)."
        )
        return updated, warning
    return legs, None
