"""FlixBus prices via the unofficial `global.api.flixbus.com` search endpoint.

This is the only long-distance operator in Sweden whose real, current fares are reachable
without a contract or a browser: the endpoint is a plain unauthenticated GET returning
JSON, verified live on 2026-07-10 (Stockholm -> Goteborg, 12 departures, 408-450 SEK).
FlixBus absorbed Swebus, so it covers most of the commercial coach network.

Two facts about the endpoint that are easy to get wrong and are pinned by tests:

* `departure_date` must be `DD.MM.YYYY`. An ISO date returns HTTP 400.
* `price.total` excludes the booking fee. `price.total_with_platform_fee` is what the
  passenger actually pays, so that is what a "cheapest trip" planner must sum.

There is an official Flix partner API, but it is gated behind a sales application. This
endpoint is the front-end's own, so it can change without notice; `parse_search` is
therefore defensive and the response contract is asserted against a recorded fixture.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

from ..models import Leg, Quote, TransportMode
from .base import HttpPriceAdapter, ore_from_major, unavailable
from .base import exact as exact_quote

SOURCE = "flixbus"
DEFAULT_BASE = "https://global.api.flixbus.com"

#: Operator strings in GTFS that this adapter is willing to price.
FLIX_OPERATORS = ("flixbus", "flix", "swebus")

#: How far a FlixBus departure may sit from the timetabled leg and still be the same bus.
#: GTFS and the sales system are separate systems and drift by a minute or two.
MATCH_TOLERANCE = timedelta(minutes=5)

#: A city's uuid is an identifier, not a price. Two weeks is conservative.
CITY_CACHE_TTL_SECONDS = 14 * 24 * 3600


def is_flixbus(operator: str | None) -> bool:
    if not operator:
        return False
    lowered = operator.lower()
    return any(token in lowered for token in FLIX_OPERATORS)


@dataclass(frozen=True, slots=True)
class FlixDeparture:
    uid: str
    departure: datetime
    arrival: datetime
    total_ore: int
    total_with_fee_ore: int
    status: str
    transfer_type: str
    seats_left: int | None

    @property
    def bookable(self) -> bool:
        return self.status == "available"

    @property
    def is_direct(self) -> bool:
        return self.transfer_type.lower() == "direct"


def format_flix_date(day: date) -> str:
    """FlixBus wants `DD.MM.YYYY`; an ISO date is rejected with HTTP 400."""
    return day.strftime("%d.%m.%Y")


def parse_search(payload: dict) -> list[FlixDeparture]:
    """Flatten the `trips[].results{uid: {...}}` structure into departures."""
    departures: list[FlixDeparture] = []
    for trip in payload.get("trips") or []:
        results = trip.get("results") or {}
        for uid, item in results.items():
            try:
                price = item["price"]
                # `total_with_platform_fee` is absent on some legacy responses.
                with_fee = price.get("total_with_platform_fee", price["total"])
                departure = datetime.fromisoformat(item["departure"]["date"])
                arrival = datetime.fromisoformat(item["arrival"]["date"])
            except (KeyError, TypeError, ValueError):
                continue
            available = item.get("available") or {}
            departures.append(
                FlixDeparture(
                    uid=item.get("uid", uid),
                    departure=departure,
                    arrival=arrival,
                    total_ore=ore_from_major(price["total"]),
                    total_with_fee_ore=ore_from_major(with_fee),
                    status=str(item.get("status", "")),
                    transfer_type=str(item.get("transfer_type", "")),
                    seats_left=available.get("seats"),
                )
            )
    departures.sort(key=lambda d: d.departure)
    return departures


def deeplink(origin_city: str, destination_city: str, day: date) -> str:
    return (
        "https://shop.flixbus.se/search"
        f"?departureCity={origin_city}&arrivalCity={destination_city}"
        f"&rideDate={format_flix_date(day)}&adult=1"
    )


class FlixBusAdapter(HttpPriceAdapter):
    """Prices a timetabled FlixBus leg by matching it to a live departure."""

    name = SOURCE
    modes = frozenset({TransportMode.BUS})

    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        *,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "tripps/0.1",
        timeout: float = 20.0,
        min_interval: float = 0.35,
        currency: str = "SEK",
        search_cache_ttl: float = 300.0,
        db=None,
    ) -> None:
        super().__init__(
            client=client, user_agent=user_agent, timeout=timeout, min_interval=min_interval
        )
        self.base_url = base_url.rstrip("/")
        self.currency = currency
        self._db = db
        self._city_cache: dict[str, str | None] = {}
        self._city_locks: dict[str, asyncio.Lock] = {}
        # One search response contains every departure that day for the city pair, so a
        # search that prices six departures needs one request, not six. The entry expires:
        # this adapter outlives a single request in the server, and FlixBus re-prices.
        self._search_cache: dict[tuple[str, str, str], tuple[float, list[FlixDeparture]]] = {}
        self._search_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self.search_cache_ttl = search_cache_ttl

    def supports(self, leg: Leg) -> bool:
        return leg.mode is TransportMode.BUS and is_flixbus(leg.operator)

    # --- upstream calls ---------------------------------------------------

    async def resolve_city(self, name: str) -> str | None:
        """Free-text place -> FlixBus city uuid. Cached; None when nothing matches.

        Pricing runs every candidate leg concurrently, and many of them name the same city.
        Without the per-name lock, a dozen coroutines all miss the cache together and fire a
        dozen identical autocomplete requests before the first answer lands.
        """
        key = name.strip().lower()
        if key in self._city_cache:
            return self._city_cache[key]

        lock = self._city_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._city_cache:  # filled while we waited
                return self._city_cache[key]

            if self._db is not None:
                hit = self._db.cache_get(f"flixbus:city:{key}")
                if hit is not None:
                    value, _stale = hit
                    self._city_cache[key] = value
                    return value

            payload = await self.get_json(
                f"{self.base_url}/search/autocomplete/cities",
                params={"q": name, "lang": "en", "country": "SE"},
                budget_key=f"{self.name}:autocomplete",
            )

            cities = payload if isinstance(payload, list) else payload.get("cities", [])
            city_id = None
            for city in cities:
                if isinstance(city, dict) and city.get("id"):
                    city_id = city["id"]
                    break
            self._city_cache[key] = city_id
            if self._db is not None:
                # City ids are stable identifiers, not prices; they can be cached for weeks.
                self._db.cache_put(f"flixbus:city:{key}", city_id, CITY_CACHE_TTL_SECONDS)
            return city_id

    async def search(
        self, from_city_id: str, to_city_id: str, day: date
    ) -> list[FlixDeparture]:
        key = (from_city_id, to_city_id, day.isoformat())
        cached = self._search_cache.get(key)
        if cached is not None:
            stored_at, departures = cached
            if time.monotonic() - stored_at < self.search_cache_ttl:
                return departures
            del self._search_cache[key]

        # Candidates are priced concurrently and mostly share a city pair; without this lock
        # they would all miss the memo at once and each send the same search.
        lock = self._search_locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._search_cache.get(key)
            if cached is not None and time.monotonic() - cached[0] < self.search_cache_ttl:
                return cached[1]
            return await self._search_uncached(key, from_city_id, to_city_id, day)

    async def _search_uncached(
        self, key: tuple[str, str, str], from_city_id: str, to_city_id: str, day: date
    ) -> list[FlixDeparture]:
        payload = await self.get_json(
            f"{self.base_url}/search/service/v4/search",
            params={
                "from_city_id": from_city_id,
                "to_city_id": to_city_id,
                "departure_date": format_flix_date(day),
                "products": '{"adult":1}',
                "currency": self.currency,
                "locale": "en_GB",
                "search_by": "cities",
                "include_after_midnight_rides": "1",
            },
        )
        if not isinstance(payload, dict):
            raise ValueError(f"expected an object, got {type(payload).__name__}")
        departures = parse_search(payload)
        self._search_cache[key] = (time.monotonic(), departures)
        return departures

    def clear_search_cache(self) -> None:
        """Drop memoized searches. The cache is per-adapter and lives as long as the process,
        so a long-running server must invalidate it rather than serve yesterday's fares."""
        self._search_cache.clear()

    # --- adapter surface --------------------------------------------------

    async def _city_id_for(self, leg: Leg, which: str) -> str | None:
        stop = leg.from_stop if which == "from" else leg.to_stop
        cached = stop.external_ids.get("flixbus_city")
        if cached:
            return cached
        return await self.resolve_city(stop.name)

    async def quote_leg(self, leg: Leg) -> Quote:
        if not self.supports(leg):
            return unavailable(self.name, None, "not a FlixBus leg")

        day = leg.departure.date()
        link = deeplink(leg.from_stop.name, leg.to_stop.name, day)

        try:
            from_id = await self._city_id_for(leg, "from")
            to_id = await self._city_id_for(leg, "to")
            if not from_id or not to_id:
                return unavailable(self.name, link, "could not resolve FlixBus city ids")

            departures = await self.search(from_id, to_id, day)
        except (httpx.HTTPError, ValueError) as exc:
            self.mark_down(f"search failed: {exc}")
            return unavailable(self.name, link, f"FlixBus lookup failed: {exc}")

        self.mark_ok()
        match = self._match(departures, leg.departure, leg.arrival)
        if match is None:
            cheapest = self._cheapest_bookable(departures)
            if cheapest is None:
                return unavailable(self.name, link, "no bookable FlixBus departure that day")
            # The timetabled bus is not on sale; the day's cheapest is a fair fallback,
            # but it is a different departure, so it is not an EXACT quote for this leg.
            from .base import estimated

            return estimated(
                self.name,
                cheapest.total_with_fee_ore,
                deeplink=link,
                note=(
                    "no FlixBus departure matched this timetabled leg; "
                    f"showing the cheapest departure on {day.isoformat()}"
                ),
            )

        if not match.bookable:
            return unavailable(self.name, link, f"departure is {match.status}")

        note = None
        if match.seats_left is not None and match.seats_left <= 5:
            note = f"only {match.seats_left} seats left at this price"
        return exact_quote(
            self.name,
            match.total_with_fee_ore,
            deeplink=link,
            fare_class="standard (incl. booking fee)",
            note=note,
        )

    def _match(
        self, departures: list[FlixDeparture], when: datetime, until: datetime | None = None
    ) -> FlixDeparture | None:
        """The departure that best fits the timetabled leg, within tolerance.

        Matching on time rather than on a shared identifier is forced on us: the GTFS feed
        and the sales system expose no common trip id.

        Departure time alone is not enough. A live Stockholm -> Goteborg response contains
        two buses leaving at 07:30, one arriving 13:45 and one 15:25, at different prices.
        Picking either would be a coin flip, so arrival time breaks the tie and the leg is
        priced as the bus it actually is.
        """
        if when.tzinfo is None:
            return None
        candidates = [
            candidate
            for candidate in departures
            if abs(candidate.departure - when) <= MATCH_TOLERANCE
        ]
        if not candidates:
            return None
        if until is None or until.tzinfo is None:
            return min(candidates, key=lambda c: abs(c.departure - when))
        return min(
            candidates,
            key=lambda c: (abs(c.arrival - until), abs(c.departure - when)),
        )

    @staticmethod
    def _cheapest_bookable(departures: list[FlixDeparture]) -> FlixDeparture | None:
        bookable = [d for d in departures if d.bookable]
        return min(bookable, key=lambda d: d.total_with_fee_ore) if bookable else None
