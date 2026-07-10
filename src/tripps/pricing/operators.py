"""Fallbacks for operators whose prices we cannot obtain.

Most Swedish long-distance rail fares are simply not reachable. They are yield-managed and
sold only through each operator's own channel; there is no public API, and the sanctioned
route (Samtrafiken's ACCESS/OSDM offer service, or SilverRail) is gated behind a reseller
contract with no self-serve path. sj.se has an internal price service, but its endpoint is
not documented anywhere and was not captured, so this project does not pretend to know it.

Two honest fallbacks, in priority order:

* `StaticFareAdapter` prices legs from a fare table the operator publishes as a fixed
  price list. Malartag and Oresundstag charge a fixed origin-destination fare that does not
  vary with when you book, so a table is a faithful representation. It ships **empty**: an
  invented number presented as a fare would be worse than no number at all. Anything it
  does return is ESTIMATED.

* `DeeplinkAdapter` is the catch-all. It cannot price a leg, so it says so, and hands the
  traveller the operator's booking site. The itinerary then carries `total_price_ore = None`
  and ranks below every fully priced option, which is exactly right: we do not know that it
  is cheap, so we must not imply that it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..interfaces import HealthState, PriceAdapter, SourceHealth
from ..models import Leg, PriceConfidence, Quote, TransportMode
from .base import estimated, unavailable

#: Where a traveller actually buys a ticket for each operator in the national feed.
#: Verified to resolve on 2026-07-10. Several sit behind a WAF that rejects scripted
#: clients, which is one more reason this adapter links out rather than scrapes.
BOOKING_URLS: dict[str, str] = {
    "SJ": "https://www.sj.se",
    "SJ Nord": "https://www.sj.se",
    "Snälltåget": "https://www.snalltaget.se",
    "Vy Tåg": "https://www.vy.se",
    "Vy Norge": "https://www.vy.se",
    "Vy Bus4You": "https://www.bus4you.se",
    "Vy flygbussarna": "https://www.flygbussarna.se",
    "VR Snabbtåg": "https://www.vr.se",
    "Tågab": "https://www.tagab.se",
    "Mälartåg": "https://www.malartag.se",
    "Öresundståg": "https://www.oresundstag.se",
    "Norrtåg": "https://www.norrtag.se",
    "Inlandsbanan": "https://www.inlandsbanan.se",
    "Y-buss": "https://www.ybuss.se",
    "MasExpressen": "https://www.masexpressen.se",
    "Härjedalingen": "https://www.harjedalingen.se",
    "Destination Gotland": "https://www.destinationgotland.se",
    "Tåg i Bergslagen": "https://www.tagibergslagen.se",
    "Krösatåg": "https://www.krosatagen.se",
    "Krösatågen": "https://www.krosatagen.se",
    "Kombardo Expressen": "https://www.kombardoexpressen.se",
    "Västervik Express": "https://www.vasterviksexpressen.se",
    "Arlanda Express": "https://www.arlandaexpress.com",
}

#: Sells Resplus through-tickets across most Swedish operators, so it is a sane place to
#: send someone whose operator we do not recognise.
FALLBACK_BOOKING_URL = "https://www.resrobot.se"


def booking_url(operator: str | None) -> str:
    if not operator:
        return FALLBACK_BOOKING_URL
    if operator in BOOKING_URLS:
        return BOOKING_URLS[operator]
    lowered = operator.casefold()
    for name, url in BOOKING_URLS.items():
        if name.casefold() in lowered or lowered in name.casefold():
            return url
    return FALLBACK_BOOKING_URL


@dataclass(frozen=True, slots=True)
class FareRow:
    operator: str
    from_stop: str
    to_stop: str
    amount_ore: int
    note: str = ""


class StaticFareAdapter(PriceAdapter):
    """Prices legs from a published, non-dynamic fare table.

    Ships empty. Populate it only from prices an operator actually publishes; every quote is
    ESTIMATED and carries the source of the number.
    """

    name = "fare-table"
    modes = frozenset({TransportMode.TRAIN, TransportMode.BUS, TransportMode.FERRY})

    def __init__(self, rows: list[FareRow] | None = None) -> None:
        self._rows: dict[tuple[str, str, str], FareRow] = {}
        for row in rows or []:
            self._rows[(row.operator, row.from_stop, row.to_stop)] = row

    @classmethod
    def from_file(cls, path: Path | str) -> StaticFareAdapter:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([FareRow(**row) for row in payload])

    def _row(self, leg: Leg) -> FareRow | None:
        return self._rows.get((leg.operator or "", leg.from_stop.id, leg.to_stop.id))

    def supports(self, leg: Leg) -> bool:
        return leg.mode in self.modes and self._row(leg) is not None

    async def quote_leg(self, leg: Leg) -> Quote:
        row = self._row(leg)
        if row is None:
            return unavailable(self.name, booking_url(leg.operator), "no fare table entry")
        return estimated(
            self.name,
            row.amount_ore,
            deeplink=booking_url(leg.operator),
            note=row.note or "Fixed published fare, not a live quote.",
        )

    async def health(self) -> SourceHealth:
        if not self._rows:
            return SourceHealth(self.name, HealthState.DISABLED, "fare table is empty")
        return SourceHealth(self.name, HealthState.OK, f"{len(self._rows)} fares")


class DeeplinkAdapter(PriceAdapter):
    """Last resort: describe the leg, link to where it is sold, price nothing.

    Placed last in the adapter list so that any source able to produce a real number wins.
    """

    name = "operator-link"
    modes = frozenset(
        {TransportMode.TRAIN, TransportMode.BUS, TransportMode.FERRY, TransportMode.LOCAL_TRANSIT}
    )

    def supports(self, leg: Leg) -> bool:
        return leg.mode in self.modes

    async def quote_leg(self, leg: Leg) -> Quote:
        operator = leg.operator or "this operator"
        if leg.mode is TransportMode.LOCAL_TRANSIT:
            note = (
                f"{operator} local transit: buy a zone ticket separately. "
                "Its price is not included in the total."
            )
        else:
            note = (
                f"{operator} does not publish prices programmatically, so this leg is "
                "unpriced. Check the fare on their site."
            )
        return Quote(
            source=self.name,
            amount_ore=None,
            confidence=PriceConfidence.UNAVAILABLE,
            deeplink=booking_url(leg.operator),
            note=note,
        )

    async def health(self) -> SourceHealth:
        return SourceHealth(
            self.name, HealthState.DISABLED, "link-out only; no price source for these operators"
        )
