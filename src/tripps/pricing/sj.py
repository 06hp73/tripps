"""SJ prices via the reverse-engineered sj.se booking API.

SJ is the dominant Swedish long-distance operator and the biggest hole in the price
picture. It publishes no official API, and the sanctioned channel (Samtrafiken ACCESS /
SilverRail) needs a reseller contract. But sj.se's own booking backend is a plain JSON API
on `prod-api.adp.sj.se`, authenticated only by a subscription key baked into the site's
JavaScript bundle - no cookie, no login, no CSRF. Verified end to end on 2026-07-10.

Two facts make this clean rather than a hack:

* **Station codes are UIC codes, which are exactly our GTFS stop ids.** `740000001` is
  Stockholm Central in both systems, so a routed leg's stop ids go straight into the API
  with no mapping table.
* **The key is discoverable and self-refreshing.** It sits in the site bundle next to
  `"Ocp-Apim-Subscription-Key"`. The adapter fetches the bundle, extracts every candidate,
  and keeps the one that a probe call actually accepts - so a rotated key is picked up
  automatically, and a hardcoded last-known-good value is only the final fallback.

The pricing chain is three calls, mirroring what the site does:

  1. POST /v3/search {origin, destination, departureDate, passengers}
       -> departureSearchId, passengerListId
  2. GET  /v3/departures/search/{departureSearchId}
       -> the day's departures, each with a departureId and times
  3. GET  /v3/departures/{departureId}/offers?passengerListId=...
       -> priceFrom.price, the cheapest fare for that departure

Steps 1-2 cover a whole day for one origin/destination and are cached per (o, d, date);
only step 3 is per departure. Prices are EXACT: they come from SJ's live sales engine for
that specific train. This remains a reverse-engineered private endpoint, so every response
is parsed defensively and pinned by a contract test against recorded fixtures.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

from ..models import Leg, Quote, TransportMode
from .base import HttpPriceAdapter, ore_from_major, unavailable
from .base import exact as exact_quote

SOURCE = "sj"
DEFAULT_API = "https://prod-api.adp.sj.se/public/sales/booking/v3"
DEFAULT_HOME = "https://www.sj.se"

#: Last-known-good booking key (2026-07-10). Only used if live extraction fails; the
#: adapter normally pulls the current key from the site bundle.
FALLBACK_KEY = "d6625619def348d38be070027fd24ff6"

#: GTFS operators SJ's consumer API sells. SJ and its Norrland brand are direct; others
#: (regional PTAs) are sold only through their own channels, so they stay on the deep-link
#: fallback rather than being mispriced here.
SJ_OPERATORS = frozenset({"SJ", "SJ Nord"})

#: How far an SJ departure may sit from the timetabled leg and still be the same train.
MATCH_TOLERANCE = timedelta(minutes=5)

#: The day search (steps 1-2) is stable for the day; cache it well inside the
#: passengerListId expiry so a multi-leg search does not re-search per leg.
DAY_CACHE_TTL_SECONDS = 900

#: Budget key for the cached day-search calls (search POST + departures GET). Kept SEPARATE
#: from the per-departure offer calls (`SOURCE`) so that setting up several origin/destination
#: pairs does not starve the price lookups - a long route touches many O/D pairs, and with one
#: shared key those setup calls exhausted the whole SJ budget before any fare was fetched.
DAY_BUDGET_KEY = f"{SOURCE}:day"

_KEY_RE = re.compile(r'Ocp-Apim-Subscription-Key["\x60]\s*:\s*["\x60]([a-f0-9]{32})["\x60]')
_BUNDLE_RE = re.compile(r'assets/(main-[A-Za-z0-9_]+\.js)')


@dataclass(frozen=True, slots=True)
class SJDeparture:
    departure_id: str
    departure: datetime
    arrival: datetime
    changes: int
    producer: str

    @property
    def is_direct(self) -> bool:
        return self.changes == 0


def parse_search(payload: dict) -> tuple[str | None, str | None]:
    """(departureSearchId, passengerListId) from the search response."""
    return payload.get("departureSearchId"), payload.get("passengerListId")


def parse_departures(payload: dict) -> list[SJDeparture]:
    """Flatten the outbound travel into departures. Ignores the return direction."""
    departures: list[SJDeparture] = []
    for travel in payload.get("travels") or []:
        for raw in travel.get("departures") or []:
            try:
                dep = datetime.fromisoformat(raw["departureDateTime"])
                arr = datetime.fromisoformat(raw["arrivalDateTime"])
                departure_id = raw["departureId"]
            except (KeyError, TypeError, ValueError):
                continue
            departures.append(
                SJDeparture(
                    departure_id=departure_id,
                    departure=dep,
                    arrival=arr,
                    changes=int(raw.get("numberOfChanges") or 0),
                    producer=str(raw.get("producer") or ""),
                )
            )
    departures.sort(key=lambda d: d.departure)
    return departures


def parse_offer_price_ore(payload: dict) -> int | None:
    """Cheapest fare for a departure, in öre, or None when it cannot be sold.

    `priceFrom.price` is the lowest purchasable fare across classes and flex levels, which
    is exactly what a cheapest-trip planner wants. It arrives as a decimal string in SEK.
    """
    if not payload.get("available"):
        return None
    price_from = payload.get("priceFrom")
    if not isinstance(price_from, dict):
        return None
    amount = price_from.get("price")
    if amount is None:
        return None
    try:
        return ore_from_major(amount)
    except (TypeError, ValueError):
        return None


def deeplink(leg: Leg) -> str:
    """SJ's own search page. Falls back to the site root, since the choose-journey URL
    keys on SJ's station display names, which need not match our GTFS names."""
    return DEFAULT_HOME


class SJAdapter(HttpPriceAdapter):
    """Prices SJ train legs from the sj.se booking backend."""

    name = SOURCE
    modes = frozenset({TransportMode.TRAIN})

    def __init__(
        self,
        api_base: str = DEFAULT_API,
        home_url: str = DEFAULT_HOME,
        *,
        operators: frozenset[str] = SJ_OPERATORS,
        key: str | None = None,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "tripps/0.1",
        timeout: float = 20.0,
        # SJ's booking backend is production infrastructure serving real ticket sales, so a
        # short interval is well within what it handles; pricing a journey is a few calls.
        min_interval: float = 0.12,
        day_cache_ttl: float = DAY_CACHE_TTL_SECONDS,
    ) -> None:
        super().__init__(
            client=client, user_agent=user_agent, timeout=timeout, min_interval=min_interval
        )
        self.api_base = api_base.rstrip("/")
        self.home_url = home_url.rstrip("/")
        self.operators = operators
        self._key = key
        self._key_lock = asyncio.Lock()
        self._day_cache: dict[tuple[str, str, str], tuple[float, str, list[SJDeparture]]] = {}
        self._day_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self.day_cache_ttl = day_cache_ttl

    def supports(self, leg: Leg) -> bool:
        return (
            leg.mode is TransportMode.TRAIN
            and (leg.operator or "") in self.operators
            and leg.from_stop.id.isdigit()
            and leg.to_stop.id.isdigit()
        )

    # --- subscription key -------------------------------------------------

    async def ensure_key(self) -> str:
        """Resolve and cache the booking key. Safe to call at startup to pre-warm it."""
        if self._key:
            return self._key
        async with self._key_lock:
            if self._key:
                return self._key
            self._key = await self._resolve_key()
            return self._key

    async def _resolve_key(self) -> str:
        """Extract the current key from the site bundle and validate it; fall back if unable.

        Key resolution deliberately does not go through `get_json`, so it never spends the
        per-search call budget: it happens once per process, not once per journey.
        """
        client = await self.http()
        try:
            home = await client.get(f"{self.home_url}/en")
            bundle_match = _BUNDLE_RE.search(home.text)
            if bundle_match:
                bundle = await client.get(f"{self.home_url}/assets/{bundle_match.group(1)}")
                for candidate in _KEY_RE.findall(bundle.text):
                    if await self._key_works(candidate):
                        return candidate
        except httpx.HTTPError as exc:
            self.mark_degraded(f"key extraction failed: {exc}")
        if await self._key_works(FALLBACK_KEY):
            return FALLBACK_KEY
        self.mark_down("no working SJ subscription key")
        return FALLBACK_KEY

    async def _key_works(self, key: str) -> bool:
        client = await self.http()
        try:
            resp = await client.get(
                f"{self.api_base}/config", headers=self._headers(key)
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _headers(self, key: str) -> dict[str, str]:
        return {
            "ocp-apim-subscription-key": key,
            "x-client-name": "sjse-booking-client",
            "content-type": "application/json",
            "accept-language": "en-GB",
            "origin": self.home_url,
            "referer": f"{self.home_url}/",
        }

    # --- the three-call chain ---------------------------------------------

    async def _day_departures(
        self, origin: str, destination: str, day: date
    ) -> tuple[str, list[SJDeparture]]:
        """(passengerListId, departures) for one origin/destination/day. Cached and locked."""
        cache_key = (origin, destination, day.isoformat())
        cached = self._day_cache.get(cache_key)
        if cached is not None and time.monotonic() - cached[0] < self.day_cache_ttl:
            return cached[1], cached[2]

        lock = self._day_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._day_cache.get(cache_key)
            if cached is not None and time.monotonic() - cached[0] < self.day_cache_ttl:
                return cached[1], cached[2]

            key = await self.ensure_key()
            search = await self._post_json(
                f"{self.api_base}/search",
                {
                    "origin": origin,
                    "destination": destination,
                    "departureDate": day.isoformat(),
                    "passengers": [{"passengerCategory": {"type": "ADULT"}}],
                },
                key,
                budget_key=DAY_BUDGET_KEY,
            )
            search_id, plid = parse_search(search)
            if not search_id or not plid:
                raise ValueError("SJ search returned no departureSearchId/passengerListId")

            listing = await self.get_json(
                f"{self.api_base}/departures/search/{search_id}",
                budget_key=DAY_BUDGET_KEY,
            )
            departures = parse_departures(listing if isinstance(listing, dict) else {})
            self._day_cache[cache_key] = (time.monotonic(), plid, departures)
            return plid, departures

    async def _post_json(self, url: str, body: dict, key: str, *, budget_key: str | None = None) -> dict:
        if self._budget is not None:
            self._budget.consume(budget_key or self.name)  # raises BudgetExceeded
        await self._limiter.acquire()
        client = await self.http()
        resp = await client.post(url, json=body, headers=self._headers(key))
        if resp.status_code == 401:
            self._key = None  # subscription key rotated; force re-resolution next search
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError(f"expected an object, got {type(payload).__name__}")
        return payload

    async def get_json(self, url: str, params: dict | None = None, *, budget_key: str | None = None):
        # SJ's GET endpoints need the subscription key header too.
        if self._budget is not None:
            self._budget.consume(budget_key or self.name)
        await self._limiter.acquire()
        client = await self.http()
        key = await self.ensure_key()
        resp = await client.get(url, params=params, headers=self._headers(key))
        if resp.status_code == 401:
            self._key = None  # subscription key rotated; force re-resolution next search
        resp.raise_for_status()
        return resp.json()

    async def _offer_price_ore(self, departure_id: str, plid: str) -> int | None:
        payload = await self.get_json(
            f"{self.api_base}/departures/{departure_id}/offers",
            params={"passengerListId": plid},
            budget_key=self.name,
        )
        return parse_offer_price_ore(payload if isinstance(payload, dict) else {})

    # --- adapter surface --------------------------------------------------

    def _match(self, departures: list[SJDeparture], leg: Leg) -> SJDeparture | None:
        """The direct SJ departure that is this timetabled train.

        A routed leg is a single train, so only direct (zero-change) departures can be the
        same service. Among those, closest departure time within tolerance wins, arrival
        breaking ties.
        """
        candidates = [
            d
            for d in departures
            if d.is_direct and abs(d.departure - leg.departure) <= MATCH_TOLERANCE
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda d: (abs(d.departure - leg.departure), abs(d.arrival - leg.arrival)),
        )

    async def quote_leg(self, leg: Leg) -> Quote:
        if not self.supports(leg):
            return unavailable(self.name, deeplink(leg), "not an SJ-sold train leg")

        link = deeplink(leg)
        try:
            plid, departures = await self._day_departures(
                leg.from_stop.id, leg.to_stop.id, leg.departure.date()
            )
        except (httpx.HTTPError, ValueError) as exc:
            self.mark_down(f"SJ day search failed: {exc}")
            return unavailable(self.name, link, f"SJ lookup failed: {exc}")

        match = self._match(departures, leg)
        if match is None:
            return unavailable(
                self.name, link, "no matching SJ departure for this timetabled train"
            )

        try:
            amount = await self._offer_price_ore(match.departure_id, plid)
        except (httpx.HTTPError, ValueError) as exc:
            self.mark_down(f"SJ offer lookup failed: {exc}")
            return unavailable(self.name, link, f"SJ price lookup failed: {exc}")

        self.mark_ok()
        if amount is None:
            return unavailable(self.name, link, "SJ departure is sold out or has no fare")
        return exact_quote(
            self.name,
            amount,
            deeplink=link,
            fare_class="cheapest available (second class)",
            note="Live SJ fare; book on sj.se.",
        )
