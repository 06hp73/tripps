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
out an unused regional period ticket. A rental is, for its window, exactly a held period card:
the marginal cost of a covered leg is again zero. So a registered rental is modelled as a
*windowed* card - covered legs price at 0 only on dates inside `[valid_from, valid_to]` - and
its period price plus a blunt warning are surfaced, because renting/reselling a period ticket
violates several PTAs' terms (Skanetrafiken, SL, Vasttrafik) and the ticket can be blocked.
Honest limit, stated like the others: the rental's period price is *not* added into a one-off
trip total, so a rental is only genuinely cheaper than single tickets if enough covered travel
falls inside its window. Tickital has no scrapable listings surface (an app-only backend), so
rentals are registered by hand from what the user sees in the app, never scanned live.
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
        if card is not None:
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
        if card.coverage_model == "operator-only":
            return True
        return bool(self._region_stops.get(card_id))

    def covers(self, card_id: str, leg: Leg) -> bool:
        card = self._cards.get(card_id)
        if card is None or leg.mode is TransportMode.WALK:
            return False
        operator = leg.operator or ""
        if operator in GLOBAL_EXCLUSIONS or operator not in card.honored_operators:
            return False
        if card.coverage_model == "operator-only":
            return True
        region = self._region_stops.get(card_id) or frozenset()
        return leg.from_stop.id in region and leg.to_stop.id in region

    def covering_card(self, held_ids, leg: Leg) -> TravelCard | None:
        """The first held card that covers this leg, or None."""
        for card_id in held_ids:
            if self.covers(card_id, leg):
                return self._cards.get(card_id)
        return None


#: Marks a quote whose zero price comes from a tickital rental rather than an owned card, so
#: the orchestrator can raise the period-cost + terms-of-service warning for it.
TICKITAL_FARE_CLASS = "tickital rental"


class PassAdapter(PriceAdapter):
    """Prices a leg at 0 when a held travel card - or an active tickital rental - covers it.

    Placed FIRST in the pricing chain: `_adapter_for` returns the first adapter whose
    `supports` is True, so a covered leg is priced free here and never reaches SJ/Tora/etc. A
    leg no held card covers is not supported, and pricing falls through to the paid adapters.

    Owned cards win over tickital rentals when both cover a leg: an owned card is free with no
    strings, a rental costs money and carries a terms warning, so the cheaper-and-cleaner note
    is preferred.
    """

    name = "travelcard"
    modes = frozenset(TransportMode)
    provides_price = True

    def __init__(self, agency_stops: dict[str, list[str]] | None = None) -> None:
        self._agency_stops = agency_stops
        self._coverage: PassCoverage | None = None
        self._coverage_key: int | None = None
        self._held: frozenset[str] = frozenset()
        self._rentals: list[TickitalRental] = []

    def prepare(self, timetable, held_ids, rentals=None) -> None:
        """Bind held cards + tickital rentals and (re)compute coverage, cached by identity.

        The Planner calls this per search. Coverage (the per-card region-stop sets) is derived
        from the whole registry and the timetable, not from which cards are held, so it is
        recomputed only when the timetable object actually changes - repeat searches stay cheap.
        """
        self._held = frozenset(held_ids or ())
        self._rentals = list(rentals or [])
        if (self._held or self._rentals) and (
            self._coverage is None or self._coverage_key != id(timetable)
        ):
            self._coverage = PassCoverage(timetable, agency_stops=self._agency_stops)
            self._coverage_key = id(timetable)

    def _card_for(self, leg: Leg) -> tuple[TravelCard, TickitalRental | None] | None:
        """The card (and, if it is a rental, the rental) that frees this leg, or None.

        Owned cards are checked first so they take precedence; then any tickital rental whose
        window contains the leg's departure date.
        """
        if self._coverage is None:
            return None
        if self._held:
            card = self._coverage.covering_card(self._held, leg)
            if card is not None:
                return (card, None)
        if self._rentals:
            day = leg.departure.date()
            for rental in self._rentals:
                if rental.active_on(day) and self._coverage.covers(rental.provider_id, leg):
                    card = self._coverage.card(rental.provider_id)
                    if card is not None:
                        return (card, rental)
        return None

    def supports(self, leg: Leg) -> bool:
        return self._card_for(leg) is not None

    async def quote_leg(self, leg: Leg) -> Quote:
        found = self._card_for(leg)
        if found is None:
            return Quote.unavailable(self.name, note="not covered by a held card")
        card, rental = found
        if rental is None:
            return Quote(
                source=self.name,
                amount_ore=0,
                confidence=PriceConfidence.EXACT,
                fare_class="travel card",
                note=f"Included with your {card.name} period ticket",
            )
        price_sek = rental.price_ore / 100
        return Quote(
            source=self.name,
            amount_ore=0,
            confidence=PriceConfidence.EXACT,
            fare_class=TICKITAL_FARE_CLASS,
            note=(
                f"Covered by a tickital {card.name} rental - {price_sek:.0f} SEK for "
                f"{rental.valid_from:%b %d}-{rental.valid_to:%b %d}. Renting a period ticket "
                f"violates {card.name}'s terms and the ticket can be blocked."
            ),
        )
