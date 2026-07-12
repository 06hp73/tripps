"""Regional-rail and Turnit-bus prices via Trainplanet's Tora reseller API.

The single biggest gap after SJ was the regional operators - Oresundstag, Malartag,
Skanetrafiken (Pagatag), Tag i Bergslagen - and the Turnit-platform coaches like Vy Bus4You.
They sell only through their own webshops, which all turn out to run on the same backend:
Trainplanet's "Tora" white-label platform. One reverse-engineered endpoint prices all of
them, verified live 2026-07-10.

`POST https://wl.tora.trainplanet.com/v1/offers` takes an origin, a destination, a date and
a passenger, and returns journeys from *every* operator Trainplanet resells for that pair -
one live response for Malmo->Goteborg carried Oresundstag (435 SEK), SJ (653) and Vy Bus4You
(224). So the adapter queries once per origin/destination/day and picks out the journey whose
carrier matches the routed leg.

Two facts make it work, and one makes it awkward:

* **Place ids are UIC codes, which are our GTFS stop ids.** `urn:x_swe:stn:740000003` is
  Malmo C; the id is built straight from `leg.from_stop.id`.
* **No authentication.** No key, no cookie, no login.
* **But a TLS-fingerprint WAF blocks non-browser clients.** A plain httpx or curl POST gets
  403 while the same request from a browser succeeds. So this adapter alone speaks through
  `primp` with Chrome impersonation, the same library the flight scraper uses, run in a
  worker thread because it is synchronous.

Prices are EXACT - they come from the operators' own live sales via Trainplanet. This is a
reverse-engineered private endpoint, so the response is parsed defensively and pinned by a
contract test against a recorded fixture. SJ and FlixBus are deliberately NOT routed here:
they have dedicated adapters that reach the operator directly, without a reseller in between.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ..models import Leg, Quote, TransportMode
from .base import BudgetExceeded, HttpPriceAdapter, unavailable
from .base import exact as exact_quote

log = logging.getLogger(__name__)

SOURCE = "tora"
OFFERS_URL = "https://wl.tora.trainplanet.com/v1/offers"
#: The Oresundstag tenant is not scoped to Oresundstag; it returns every operator Trainplanet
#: resells. Verified to surface Malartag, Skanetrafiken, Vy Bus4You and more.
REFERER = "https://boka.oresundstag.se"

#: GTFS operator names this adapter will try to price, matched case-insensitively against the
#: carrier Tora returns. SJ and FlixBus are excluded on purpose - their own adapters are
#: closer to the source. Buses are included because the Turnit coaches sell through Tora too.
TORA_OPERATORS = frozenset(
    {
        "Öresundståg",
        "Mälartåg",
        "Skånetrafiken",
        "Tåg i Bergslagen",
        "Snälltåget",
        "Vy Bus4You",
        "Vy Tåg",
        "Krösatåg",
        "Krösatågen",
        "Kombardo Expressen",
        "Tågab",
        "Norrtåg",
    }
)

#: How far a Tora departure may sit from the timetabled leg and still be the same service.
MATCH_TOLERANCE = timedelta(minutes=5)

#: One offers response covers a whole day for an origin/destination; regional fares are fixed
#: rather than yield-managed, so a longer cache is safe.
DAY_CACHE_TTL_SECONDS = 1800

_IMPERSONATE = "chrome_145"


def place_urn(stop_id: str) -> str:
    """`"740000003"` -> `"urn:x_swe:stn:740000003"`. The GTFS id is the UIC code Tora wants."""
    return f"urn:x_swe:stn:{stop_id}"


def _norm(name: str | None) -> str:
    return (name or "").casefold().strip()


def carrier_matches(leg_operator: str | None, tora_carrier: str | None) -> bool:
    """Is this Tora journey the same operator as the routed leg?

    Names differ in case and spacing between the feed and the reseller ("Vy Bus4You" vs
    "Vy bus4you"), so the comparison is case-insensitive and accepts a containment either
    way to absorb minor branding differences.
    """
    a, b = _norm(leg_operator), _norm(tora_carrier)
    if not a or not b:
        return False
    return a == b or a in b or b in a


@dataclass(frozen=True, slots=True)
class ToraJourney:
    departure: datetime
    arrival: datetime
    changes: int
    carriers: tuple[str, ...]
    cheapest_ore: int | None

    @property
    def is_direct(self) -> bool:
        return self.changes == 0

    @property
    def sole_carrier(self) -> str | None:
        return self.carriers[0] if len(self.carriers) == 1 else None


def _cheapest_offer_ore(journey: dict) -> int | None:
    """Lowest purchasable fare across all booking options, in öre.

    Amounts arrive already in minor units (43500 == 435.00 SEK). Zero-valued entries (fees,
    fully-discounted add-ons) are ignored so they cannot masquerade as a free ticket.
    """
    best: int | None = None
    view = journey.get("bookingOptions", {}).get("detailedBookingView", {})
    for option in view.get("bookingOptions", []):
        for offer in option.get("offers", []):
            price = offer.get("price") or {}
            amount = price.get("amount")
            if isinstance(amount, (int, float)) and amount > 0 and (best is None or amount < best):
                best = int(amount)
    return best


def parse_offers(payload: dict) -> list[ToraJourney]:
    journeys: list[ToraJourney] = []
    for raw in payload.get("journeys") or []:
        trip = raw.get("trip") or {}
        try:
            departure = datetime.fromisoformat(trip["departureTime"])
            arrival = datetime.fromisoformat(trip["arrivalTime"])
        except (KeyError, TypeError, ValueError):
            continue
        carriers = tuple(
            c.get("name") for c in trip.get("carriers", []) if isinstance(c, dict) and c.get("name")
        )
        journeys.append(
            ToraJourney(
                departure=departure,
                arrival=arrival,
                changes=int(trip.get("changes") or 0),
                carriers=carriers,
                cheapest_ore=_cheapest_offer_ore(raw),
            )
        )
    journeys.sort(key=lambda j: j.departure)
    return journeys


def deeplink(leg: Leg) -> str:
    return "https://www.oresundstag.se" if leg.mode is TransportMode.TRAIN else REFERER


class ToraAdapter(HttpPriceAdapter):
    """Prices regional-rail and Turnit-bus legs from Trainplanet's Tora reseller API."""

    name = SOURCE
    modes = frozenset({TransportMode.TRAIN, TransportMode.BUS})

    def __init__(
        self,
        offers_url: str = OFFERS_URL,
        *,
        operators: frozenset[str] = TORA_OPERATORS,
        user_agent: str = "tripps/0.1",
        min_interval: float = 0.2,
        day_cache_ttl: float = DAY_CACHE_TTL_SECONDS,
        currency: str = "SEK",
        # Bound per-request latency. Tora normally answers in ~1-2 s; without a cap, primp's
        # own default let a slow request hang ~30 s, which both flaked the canary and could
        # eat most of a search's phase-2 deadline on one leg. 15 s is generous but bounded.
        timeout: float = 15.0,
    ) -> None:
        super().__init__(user_agent=user_agent, min_interval=min_interval, timeout=timeout)
        self.offers_url = offers_url
        self.operators = operators
        self.currency = currency
        self.day_cache_ttl = day_cache_ttl
        self._day_cache: dict[tuple[str, str, str], tuple[float, list[ToraJourney]]] = {}
        self._day_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        # A primp client, not the base class's httpx one: this adapter needs browser TLS
        # impersonation to clear the WAF, which httpx cannot provide.
        self._primp = None

    def supports(self, leg: Leg) -> bool:
        return (
            leg.mode in self.modes
            and leg.from_stop.id.isdigit()
            and leg.to_stop.id.isdigit()
            and any(carrier_matches(leg.operator, op) for op in self.operators)
        )

    # --- HTTP via primp (browser impersonation, in a thread) --------------

    def _fetch_offers_sync(self, origin: str, destination: str, when: datetime) -> dict:
        from primp import Client  # noqa: PLC0415

        if self._primp is None:
            self._primp = Client(
                impersonate=_IMPERSONATE, impersonate_os="macos", referer=True,
                timeout=self._timeout,
            )
        body = {
            "currency": self.currency,
            "promotionCodes": [],
            "corporateCodes": [],
            "passengerSpecifications": [{"type": "ADULT", "externalRef": "ADULT-0"}],
            "interrail": False,
            "originPlace": place_urn(origin),
            "destinationPlace": place_urn(destination),
            "departureDateTime": when.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        resp = self._primp.post(
            self.offers_url,
            headers={
                "content-type": "application/json",
                "accept": "application/json",
                "origin": REFERER,
                "referer": f"{REFERER}/",
            },
            content=json.dumps(body).encode("utf-8"),
        )
        if resp.status_code != 200:
            raise ValueError(f"tora offers returned HTTP {resp.status_code}")
        return json.loads(resp.text)

    async def _day_offers(
        self, origin: str, destination: str, day: date, near: datetime
    ) -> list[ToraJourney]:
        cache_key = (origin, destination, day.isoformat())
        cached = self._day_cache.get(cache_key)
        if cached is not None and time.monotonic() - cached[0] < self.day_cache_ttl:
            return cached[1]

        lock = self._day_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._day_cache.get(cache_key)
            if cached is not None and time.monotonic() - cached[0] < self.day_cache_ttl:
                return cached[1]
            if self._budget is not None:
                self._budget.consume(self.name)  # raises BudgetExceeded
            await self._limiter.acquire()
            payload = await asyncio.to_thread(
                self._fetch_offers_sync, origin, destination, near
            )
            journeys = parse_offers(payload)
            self._day_cache[cache_key] = (time.monotonic(), journeys)
            return journeys

    # --- adapter surface --------------------------------------------------

    def _match(self, journeys: list[ToraJourney], leg: Leg) -> ToraJourney | None:
        """The direct, single-carrier journey that is this timetabled leg."""
        candidates = [
            j
            for j in journeys
            if j.is_direct
            and carrier_matches(leg.operator, j.sole_carrier)
            and abs(j.departure - leg.departure) <= MATCH_TOLERANCE
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda j: abs(j.departure - leg.departure))

    async def quote_leg(self, leg: Leg) -> Quote:
        if not self.supports(leg):
            return unavailable(self.name, deeplink(leg), "not a Tora-sold operator")

        link = deeplink(leg)
        try:
            journeys = await self._day_offers(
                leg.from_stop.id, leg.to_stop.id, leg.departure.date(), leg.departure
            )
        except BudgetExceeded as exc:
            # Out of calls for this search; the source is healthy, we just stopped asking.
            return unavailable(self.name, link, str(exc))
        except Exception as exc:  # noqa: BLE001 - a reseller endpoint may WAF or vanish
            self.mark_down(f"Tora lookup failed: {exc}")
            return unavailable(self.name, link, f"Tora lookup failed: {exc}")

        self.mark_ok()
        match = self._match(journeys, leg)
        if match is None:
            return unavailable(
                self.name, link, "no matching Tora journey for this timetabled service"
            )
        if match.cheapest_ore is None:
            return unavailable(self.name, link, "Tora journey has no purchasable fare")

        return exact_quote(
            self.name,
            match.cheapest_ore,
            deeplink=link,
            fare_class="cheapest available",
            note=f"Live {match.sole_carrier} fare via Trainplanet; book on the operator's site.",
        )

    async def aclose(self) -> None:
        # primp's Client has no async close; drop the reference and let the base class close
        # its (unused) httpx client.
        self._primp = None
        await super().aclose()
