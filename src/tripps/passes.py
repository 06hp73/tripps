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
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
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

    def __init__(self, timetable, cards: dict[str, TravelCard] | None = None) -> None:
        self._cards = cards or load_cards()
        by_agency: dict[str, set[str]] = defaultdict(set)
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


class PassAdapter(PriceAdapter):
    """Prices a leg at 0 when a held travel card covers it.

    Placed FIRST in the pricing chain: `_adapter_for` returns the first adapter whose
    `supports` is True, so a covered leg is priced free here and never reaches SJ/Tora/etc. A
    leg no held card covers is not supported, and pricing falls through to the paid adapters.
    """

    name = "travelcard"
    modes = frozenset(TransportMode)
    provides_price = True

    def __init__(self) -> None:
        self._coverage: PassCoverage | None = None
        self._coverage_key: int | None = None
        self._held: frozenset[str] = frozenset()

    def prepare(self, timetable, held_ids) -> None:
        """Bind the held cards and (re)compute coverage for this timetable, cached by identity.

        The Planner calls this per search. Recomputing the region-stop sets only when the
        timetable object actually changes keeps repeat searches cheap.
        """
        self._held = frozenset(held_ids or ())
        if self._held and (self._coverage is None or self._coverage_key != id(timetable)):
            self._coverage = PassCoverage(timetable)
            self._coverage_key = id(timetable)

    def _card_for(self, leg: Leg) -> TravelCard | None:
        if not self._held or self._coverage is None:
            return None
        return self._coverage.covering_card(self._held, leg)

    def supports(self, leg: Leg) -> bool:
        return self._card_for(leg) is not None

    async def quote_leg(self, leg: Leg) -> Quote:
        card = self._card_for(leg)
        if card is None:
            return Quote.unavailable(self.name, note="not covered by a held card")
        return Quote(
            source=self.name,
            amount_ore=0,
            confidence=PriceConfidence.EXACT,
            fare_class="travel card",
            note=f"Included with your {card.name} period ticket",
        )
