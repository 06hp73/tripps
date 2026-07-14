"""Regional travel cards (period tickets) that make covered legs free.

A Swedish regional period ticket (månadskort) is unlimited travel within its area, so for a
holder the marginal cost of a covered leg is zero. This models that: register the cards you
hold, and covered legs price at 0.

The load-bearing fact, verified against every PTA's own ticket pages: **the operator name is
never enough.** The same consortium-train operator - `Öresundståg`, `Krösatåg`/`Krösatågen`,
`Mälartåg`, `Tåg i Bergslagen`, `Norrtåg` - is honored by many different cards, each only
within its own region. A Skånetrafiken card frees Öresundståg inside Skåne but NOT the
onward Malmö→Göteborg leg. So a card that honors a consortium operator gates it on the
card's own served stops: a leg is free only if *both* endpoints lie in the card's region.

Coverage is deliberately conservative - it always prefers to charge over wrongly zeroing a
leg. Two models per card:

* `operator-only` - the card's own agency is region-local (SL), so any leg by that operator
  is covered without a stop check.
* `region-stops` - a consortium operator, free only if both stops are in the card's region
  (the stops served by the card's home agency in the loaded timetable).

Known v1 limits, surfaced rather than hidden: only the FULL-region (all-zone) card variant
is assumed - a partial-zone ticket covers fewer stops; SJ is never auto-freed because the
`SJ` agency mixes SJ Regional (sometimes covered) with SJ high-speed (never); the Arlanda C
passage fee is not modelled; and cross-region passes (Movingo, Norrlandsresan) are listed
but not yet honored.

**Tickital rentals.** Tickital (tickital.com) is a peer-to-peer marketplace where people rent
out an unused regional period ticket. A rental is NOT an owned pass - it costs money - so it
is not zeroed here. It is priced as a *coupon*: `TickitalAdapter` (after the paid sources)
gives a covered leg the rental's period price only when no real fare source can, and the
orchestrator's itinerary-level reconciliation then charges the rental's period price ONCE for
the covered legs of a journey, but only when that beats buying single tickets for them (or
when a covered leg has no purchasable single at all). Its terms risk is surfaced bluntly,
because renting/reselling a period ticket violates several PTAs' terms (Skanetrafiken, SL,
Vasttrafik) and the ticket can be blocked. The period cost is a one-time cost: code that sums
more than one itinerary (round trip, fare calendar) dedups it by `Quote.coupon_rental_id`.
Tickital has no scrapable listings surface (an app-only backend), so rentals are registered by
hand from what the user sees in the app, never scanned live.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

from .interfaces import PriceAdapter
from .models import Leg, PriceConfidence, Quote, TransportMode

REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "travelcards.json"

#: Operators no period card ever frees, even if a card's honored list names them. The
#: registry is already curated to exclude these, but the gate is enforced here too so a data
#: edit can never accidentally zero a premium ticket.
GLOBAL_EXCLUSIONS = frozenset(
    {
        "SJ",
        "SJ Nord",
        "Snälltåget",
        "Arlanda Express",
        "Destination Gotland",
        "Flixbus",
        "Waxholmsbolaget",
        "Vy Bus4You",
        "Vy flygbussarna",
        "MasExpressen",
        "Y-buss",
        "Härjedalingen",
        "Kombardo Expressen",
        "Västervik Express",
    }
)


@dataclass(frozen=True, slots=True)
class TravelCard:
    id: str
    name: str
    region: str
    honored_operators: frozenset[str]
    coverage_model: str  # "operator-only" | "region-stops"
    region_agencies: frozenset[str]
    coverage_note: str = ""
    confidence: str = "reported-secondhand"
    #: Curated named boundary stops this card is explicitly valid to across a county line, by
    #: GTFS stop_id. A leg with one endpoint in region and the other in this set is covered.
    #: Empty by default: cross-border validity is negotiated per PTA-pair, not a general "one
    #: stop over" rule, so nothing extends unless a verified stop is listed. Note that the
    #: region set already includes any boundary station the card's OWN agency runs to, so this
    #: is only for stations served solely by a neighbour's agency under a named agreement.
    border_stops: frozenset[str] = frozenset()
    #: Stops this card is NOT valid at even though its operator/agency serves them - the
    #: over-coverage escape hatch. Chiefly for operator-only cards (SL) whose vehicle crosses
    #: the ticket boundary: SL pendeltag 40 runs to Knivsta and Uppsala C, which need a UL
    #: fare. A leg touching a denied stop is never freed by this card.
    denied_stops: frozenset[str] = frozenset()
    #: The under-coverage escape hatch, mirror of `denied_stops`: stations the card IS valid at
    #: that the region derivation misses because the PTA's OWN feed vehicles never call there.
    #: Hallandstrafiken is the canonical case - its buses serve town stops while every train at
    #: Halmstad C runs under a train operator's agency, so the county's main station is absent
    #: from the agency-derived region. Full region members (unlike `border_stops`, which are
    #: endpoint-only extensions): they count for single-card and combined coverage alike.
    extra_region_stops: frozenset[str] = frozenset()
    #: A per-relation / partial product (Movingo, Norrlandsresan) whose exact validity is not a
    #: region polygon and cannot be verified from the feed. Such a card is registered and shown
    #: but NEVER auto-frees a leg - modelling its region as a multi-county union would over-cover
    #: a holder who only bought one relation. Priced as paid with a hint until relation modelling.
    variant_gated: bool = False


@dataclass(frozen=True, slots=True)
class TickitalRental:
    """A second-hand period ticket rented via tickital, valid over a date window.

    Within `[valid_from, valid_to]` it covers the same legs as holding `provider_id`'s card
    (same region gate), so covered legs price at 0. `price_ore` is the rental's period cost,
    surfaced with a warning rather than summed into the trip - see the module docstring.
    """

    provider_id: str
    price_ore: int
    valid_from: date
    valid_to: date
    note: str = ""
    id: int | None = None

    def active_on(self, day: date) -> bool:
        return self.valid_from <= day <= self.valid_to

    @classmethod
    def from_row(cls, row) -> TickitalRental:
        return cls(
            id=row["id"],
            provider_id=row["provider_id"],
            price_ore=row["price_ore"],
            valid_from=date.fromisoformat(row["valid_from"]),
            valid_to=date.fromisoformat(row["valid_to"]),
            note=row["note"] or "",
        )


def honored_operators(card_ids) -> frozenset[str]:
    """Operators any of the given held cards frees, for zeroing the router's price floors.

    Used by pass-aware routing: the router optimizes a price lower bound, and zero is a valid
    lower bound for a leg a held card might cover, so honored operators get a floor of 0 and
    the router stops treating covered legs as expensive. Globally-excluded (premium) operators
    are removed, since no card frees them.
    """
    cards = load_cards()
    ops: set[str] = set()
    for card_id in card_ids or ():
        card = cards.get(card_id)
        if card is not None and not card.variant_gated:
            ops |= card.honored_operators
    return frozenset(ops - GLOBAL_EXCLUSIONS)


@lru_cache(maxsize=1)
def load_cards() -> dict[str, TravelCard]:
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    cards: dict[str, TravelCard] = {}
    for entry in raw:
        cards[entry["id"]] = TravelCard(
            id=entry["id"],
            name=entry["name"],
            region=entry["region"],
            honored_operators=frozenset(entry.get("honored_operators", [])),
            coverage_model=entry.get("coverage_model", "region-stops"),
            region_agencies=frozenset(entry.get("region_agencies", [])),
            coverage_note=entry.get("coverage_note", ""),
            confidence=entry.get("confidence", "reported-secondhand"),
            border_stops=frozenset(entry.get("border_stops", [])),
            denied_stops=frozenset(entry.get("denied_stops", [])),
            extra_region_stops=frozenset(entry.get("extra_region_stops", [])),
            variant_gated=bool(entry.get("variant_gated", False)),
        )
    return cards


class PassCoverage:
    """Answers "does card X make this leg free", given a timetable to derive regions from.

    The per-card region-stop set is the stops served by the card's home agency in the loaded
    national timetable. It is computed once here (a few hundred thousand integer inserts) and
    reused, because it is stable across service dates.
    """

    def __init__(
        self,
        timetable,
        cards: dict[str, TravelCard] | None = None,
        agency_stops: dict[str, list[str]] | None = None,
    ) -> None:
        self._cards = cards or load_cards()
        # Prefer the full-feed agency->stops map when supplied: it includes county PTAs whose
        # only routes are local buses (absent from the intercity routing timetable), which is
        # exactly what defines those cards' regions. Otherwise derive from the timetable's own
        # routes (enough for the intercity PTAs, and all a test needs).
        by_agency: dict[str, set[str]] = defaultdict(set)
        if agency_stops is not None:
            for agency, stop_ids in agency_stops.items():
                by_agency[agency] = set(stop_ids)
        else:
            for route in timetable.routes:
                operator = route.info.operator or ""
                for stop_index in route.stops:
                    by_agency[operator].add(timetable.stops[stop_index].id)
        self._region_stops: dict[str, frozenset[str]] = {}
        for card in self._cards.values():
            stops: set[str] = set()
            for agency in card.region_agencies:
                stops |= by_agency.get(agency, set())
            # Curated stations the card is valid at but the PTA's own vehicles never serve
            # (Halmstad C for Hallandstrafiken: only train agencies call at the rail parent).
            stops |= card.extra_region_stops
            self._region_stops[card.id] = frozenset(stops)

    def card(self, card_id: str) -> TravelCard | None:
        return self._cards.get(card_id)

    def is_supported(self, card_id: str) -> bool:
        """A card is usable if it is operator-only, or its region resolved to some stops.

        Cards whose home agency is absent from the intercity feed resolve to an empty region
        and cannot free anything yet; they are still listed, so the user sees them, but they
        are marked unsupported rather than silently doing nothing.
        """
        card = self._cards.get(card_id)
        if card is None:
            return False
        if card.variant_gated:
            return False  # registered and listed, but hint-only, so mark it unsupported
        if card.coverage_model == "operator-only":
            return True
        return bool(self._region_stops.get(card_id))

    def covers(self, card_id: str, leg: Leg) -> bool:
        card = self._cards.get(card_id)
        if card is None or leg.mode is TransportMode.WALK:
            return False
        if card.variant_gated:
            return False  # per-relation product; never auto-freed (see TravelCard.variant_gated)
        operator = leg.operator or ""
        if operator in GLOBAL_EXCLUSIONS or operator not in card.honored_operators:
            return False
        if card.denied_stops and (
            leg.from_stop.id in card.denied_stops or leg.to_stop.id in card.denied_stops
        ):
            return False  # the card's vehicle serves this stop but the ticket is not valid there
        if card.coverage_model == "operator-only":
            return True
        region = self._region_stops.get(card_id) or frozenset()
        frm, to = leg.from_stop.id, leg.to_stop.id
        if frm in region and to in region:
            return True
        # Named cross-border extension: exactly one endpoint in region and the other a curated
        # boundary stop this card is explicitly valid to. A leg with NEITHER endpoint in region
        # lies entirely in the neighbour and is never freed. `border_stops` is empty for every
        # card by default, so this branch changes nothing until a verified stop is listed.
        border = card.border_stops
        if border and (
            (frm in region and to in border) or (to in region and frm in border)
        ):
            return True
        return False

    def covering_card(self, held_ids, leg: Leg) -> TravelCard | None:
        """The first held card that covers this leg, or None."""
        for card_id in held_ids:
            if self.covers(card_id, leg):
                return self._cards.get(card_id)
        return None

    def partially_covers(self, card_id: str, leg: Leg) -> bool:
        """True when a held card honors this leg's operator and EXACTLY ONE endpoint is in the
        card's region, but the leg is not fully covered.

        That is a leg crossing the card's county border: most Swedish PTAs let a period ticket
        span the border only by paying for the extra zones (zone-combination), which is not
        modelled here, so the leg is charged in full. This flags that the shown fare may be
        reducible with the card - an honest hint, never a discount.
        """
        card = self._cards.get(card_id)
        if card is None or leg.mode is TransportMode.WALK:
            return False
        operator = leg.operator or ""
        if operator in GLOBAL_EXCLUSIONS or operator not in card.honored_operators:
            return False
        if card.coverage_model != "region-stops":
            return False  # operator-only cards have no zone/border concept
        if self.covers(card_id, leg):
            return False  # already free (in-region, or a seeded border stop)
        region = self._region_stops.get(card_id) or frozenset()
        return (leg.from_stop.id in region) != (leg.to_stop.id in region)

    def combined_cover(self, held_ids, leg: Leg) -> list[TravelCard] | None:
        """The 2+ held cards whose regions JOINTLY cover every stop this leg calls at, or None.

        A through-train can cross several card regions - Öresundståg Lund->Göteborg runs Skåne,
        Halland, Västra Götaland. If the union of held cards that honor the operator covers the
        whole path, the marginal cost is zero. This checks the leg's full stop sequence
        (`via_stop_ids`), not just its endpoints: without the path the middle cannot be
        verified, so it returns None (fail closed, never over-covering). Single-card coverage is
        handled by `covers`; this only reports genuinely combined (2+ contributing) coverage.
        """
        if leg.mode is TransportMode.WALK or not leg.via_stop_ids:
            return None
        operator = leg.operator or ""
        if operator in GLOBAL_EXCLUSIONS:
            return None
        honoring = [
            card
            for card_id in held_ids
            if (card := self._cards.get(card_id)) is not None
            and operator in card.honored_operators
            and card.coverage_model == "region-stops"
            and not card.variant_gated
        ]
        if len(honoring) < 2:
            return None
        # Region stops ONLY - border_stops stay out of the union on purpose. A border stop is a
        # named single-card extension ("this card is valid TO that one neighbouring station"),
        # valid only as an endpoint adjacent to the card's own region. Pooling them as ordinary
        # covered stops let two cards' border extensions bridge a mid-path hop neither card
        # covers (x-trafik + dalatrafik would jointly "cover" Gävle->Örebro via a gap both
        # merely border), silently freeing a paid stretch. Cross-region handoffs still combine
        # where the regions genuinely meet, via real region stops.
        covered_by = {
            card.id: self._region_stops.get(card.id, frozenset()) - card.denied_stops
            for card in honoring
        }
        path = set(leg.via_stop_ids)
        if not path <= set().union(*covered_by.values()):
            return None
        # Only the cards that actually contribute a stop on this path, for an honest note.
        contributing = [card for card in honoring if covered_by[card.id] & path]
        return contributing if len(contributing) >= 2 else None


#: Marks a quote produced by a tickital rental (the coupon-charged leg and its zeroed
#: siblings), so the orchestrator can raise the period-cost + terms-of-service warning.
TICKITAL_FARE_CLASS = "tickital rental"

#: Quote.source for the tickital adapter. Distinct from PassAdapter's "travelcard" so the
#: coupon can tell a rental fallback from an owned-card zero, and so floor-violation checks
#: skip it the same way they skip owned cards.
TICKITAL_SOURCE = "tickital"


class PassAdapter(PriceAdapter):
    """Prices a leg at 0 when a held (owned) travel card covers it.

    Placed FIRST in the pricing chain: `_adapter_for` returns the first adapter whose
    `supports` is True, so a covered leg is priced free here and never reaches SJ/Tora/etc. A
    leg no held card covers is not supported, and pricing falls through to the paid adapters.

    Tickital rentals are NOT handled here - they are a paid second-hand ticket, not an owned
    pass, so they go through `TickitalAdapter` (placed after the paid sources) and an
    itinerary-level coupon, which charges the rental's period price once only when it beats
    buying singles. An owned card, by contrast, is genuinely free for the holder, so it zeroes
    the leg unconditionally here.
    """

    name = "travelcard"
    modes = frozenset(TransportMode)
    provides_price = True

    def __init__(self, agency_stops: dict[str, list[str]] | None = None) -> None:
        self._agency_stops = agency_stops
        self._coverage: PassCoverage | None = None
        self._coverage_key: int | None = None
        self._held: frozenset[str] = frozenset()

    def prepare(self, timetable, held_ids) -> None:
        """Bind the held cards and (re)compute coverage for this timetable, cached by identity.

        Coverage (the per-card region-stop sets) is derived from the whole registry and the
        timetable, not from which cards are held, so it is recomputed only when the timetable
        object actually changes - repeat searches stay cheap.
        """
        self._held = frozenset(held_ids or ())
        if self._held and (self._coverage is None or self._coverage_key != id(timetable)):
            self._coverage = PassCoverage(timetable, agency_stops=self._agency_stops)
            self._coverage_key = id(timetable)

    def _card_for(self, leg: Leg) -> TravelCard | None:
        if not self._held or self._coverage is None:
            return None
        return self._coverage.covering_card(self._held, leg)

    def _combined_for(self, leg: Leg) -> list[TravelCard] | None:
        """The held cards that jointly cover this through-leg, when no single card does."""
        if not self._held or self._coverage is None:
            return None
        return self._coverage.combined_cover(self._held, leg)

    def partial_card(self, leg: Leg) -> TravelCard | None:
        """A held card that partially covers this leg (crosses its county border), for a hint.

        Used by the orchestrator to tell the traveller their card may reduce a full-fare leg
        via a zone-combination add-on we do not price. Returns None when no held card applies.
        """
        if not self._held or self._coverage is None:
            return None
        for card_id in self._held:
            if self._coverage.partially_covers(card_id, leg):
                return self._coverage.card(card_id)
        return None

    def supports(self, leg: Leg) -> bool:
        return self._card_for(leg) is not None or self._combined_for(leg) is not None

    async def quote_leg(self, leg: Leg) -> Quote:
        card = self._card_for(leg)
        if card is not None:
            return Quote(
                source=self.name,
                amount_ore=0,
                confidence=PriceConfidence.EXACT,
                fare_class="travel card",
                note=f"Included with your {card.name} period ticket",
            )
        combined = self._combined_for(leg)
        if combined is not None:
            names = " + ".join(c.name for c in combined)
            return Quote(
                source=self.name,
                amount_ore=0,
                confidence=PriceConfidence.EXACT,
                fare_class="travel card",
                note=f"Included with your {names} cards, which jointly cover this journey",
            )
        return Quote.unavailable(self.name, note="not covered by a held card")


def tickital_note(card: TravelCard, rental: TickitalRental) -> str:
    """The honest one-line note carried on a rental-priced leg (and dedup key for warnings).

    Kept byte-identical between the adapter's fallback quote and the coupon's rewrite so the
    orchestrator's warning collapses to exactly one line. Says the rental is charged once for
    the covered legs, and states the terms risk - it does not claim it beat singles, because
    the same string is used both when it did and when it was the only priceable option.
    """
    price_sek = rental.price_ore / 100
    return (
        f"Covered by a tickital {card.name} rental: {price_sek:.0f} SEK for "
        f"{rental.valid_from:%b %d}-{rental.valid_to:%b %d}, charged once for the covered "
        f"legs. Renting or reselling a period ticket violates {card.name}'s terms and the "
        f"ticket can be blocked."
    )


class TickitalAdapter(PriceAdapter):
    """Fallback price for a leg an active tickital rental covers.

    Placed LAST before the deeplink adapter: a covered leg on an operator that HAS a paid
    source (e.g. Oresundstag via Tora) is priced by that source first, so the itinerary-level
    coupon can compare the rental against the real single fare. Only a covered leg no paid
    source can price (Skanetrafiken, Pagatag) reaches this adapter, which prices it at the
    rental's period price so the itinerary stays priceable instead of being dropped. The
    coupon in the orchestrator then rewrites the covered legs to charge the rental once.
    """

    name = TICKITAL_SOURCE
    modes = frozenset(TransportMode)
    provides_price = True

    def __init__(self, agency_stops: dict[str, list[str]] | None = None) -> None:
        self._agency_stops = agency_stops
        self._coverage: PassCoverage | None = None
        self._coverage_key: int | None = None
        self._rentals: list[TickitalRental] = []

    def prepare(self, timetable, rentals) -> None:
        self._rentals = list(rentals or [])
        if self._rentals and (self._coverage is None or self._coverage_key != id(timetable)):
            self._coverage = PassCoverage(timetable, agency_stops=self._agency_stops)
            self._coverage_key = id(timetable)

    def has_rentals(self) -> bool:
        return bool(self._rentals)

    def rental_for(self, leg: Leg) -> tuple[TravelCard, TickitalRental] | None:
        """The first active rental whose region covers this leg's date and endpoints, or None."""
        if self._coverage is None or not self._rentals:
            return None
        day = leg.departure.date()
        for rental in self._rentals:
            if rental.active_on(day) and self._coverage.covers(rental.provider_id, leg):
                card = self._coverage.card(rental.provider_id)
                if card is not None:
                    return (card, rental)
        return None

    def supports(self, leg: Leg) -> bool:
        return self.rental_for(leg) is not None

    async def quote_leg(self, leg: Leg) -> Quote:
        found = self.rental_for(leg)
        if found is None:
            return Quote.unavailable(self.name, note="no active rental covers this leg")
        card, rental = found
        return Quote(
            source=self.name,
            amount_ore=rental.price_ore,
            confidence=PriceConfidence.EXACT,
            fare_class=TICKITAL_FARE_CLASS,
            note=tickital_note(card, rental),
            coupon_rental_id=rental.id,
        )
