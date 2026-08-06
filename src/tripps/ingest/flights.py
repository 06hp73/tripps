"""Domestic flight offers, scraped from Google Flights.

There is no usable flight-price API for a project this size. Amadeus Self-Service, the
obvious choice, deactivates its keys on 2026-07-17. Kiwi's Tequila is invitation-only,
Skyscanner is partner-only, and Duffel does not carry SAS. That leaves Google Flights,
read through the `fast-flights` package, which decodes the protobuf query parameter its
front end uses.

Two things this module gets right that a naive integration does not:

**The EU consent wall.** Requesting google.com/travel/flights from a Swedish IP returns a
`consent.google.com` interstitial, not a flight page. `fast-flights` does not handle it, so
its parser dies on a missing `<script>`. A consent cookie is sent with the request, which is
the programmatic equivalent of clicking through the banner once.

**Foreign connections.** A Stockholm-to-Goteborg search returns itineraries via Frankfurt
and Helsinki. This planner routes strictly within Sweden, so anything that touches a
non-Swedish airport, or connects at all, is discarded.

Offers arrive already priced, so the synthetic trips they produce carry an exact fare and
need no price floor. Phase-2 pricing reads that fare back rather than searching again.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from ..models import SEK, STOCKHOLM_TZ, TransportMode
from ..routing.timetable import RouteAddition, RouteInfo, Trip
from ..timeutil import to_service_seconds
from .airports import Airport, is_swedish

log = logging.getLogger(__name__)

TZ = ZoneInfo(STOCKHOLM_TZ)
FLIGHT_OPERATOR_PREFIX = "air"

#: Google serves an interstitial to EU visitors until a consent choice is stored. This is
#: a standard "socs" consent cookie; without it the response is a consent page, not flights.
GOOGLE_CONSENT_COOKIE = {
    "SOCS": "CAISNQgQEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjQwMzA1LjA2X3AwGgJlbiADGgYIgLC_rwY"
}

#: Flight prices move, but not minute to minute, and each lookup is a full page scrape.
FLIGHT_CACHE_TTL_SECONDS = 3600


@dataclass(frozen=True, slots=True)
class FlightOffer:
    carrier: str
    from_iata: str
    to_iata: str
    departure: datetime
    arrival: datetime
    price_ore: int

    @property
    def route_id(self) -> str:
        return f"air:{self.from_iata}-{self.to_iata}:{self.departure:%H%M}"

    @property
    def duration_seconds(self) -> int:
        return int((self.arrival - self.departure).total_seconds())

    def to_dict(self) -> dict:
        return {
            "carrier": self.carrier,
            "from_iata": self.from_iata,
            "to_iata": self.to_iata,
            "departure": self.departure.isoformat(),
            "arrival": self.arrival.isoformat(),
            "price_ore": self.price_ore,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> FlightOffer:
        return cls(
            carrier=raw["carrier"],
            from_iata=raw["from_iata"],
            to_iata=raw["to_iata"],
            departure=datetime.fromisoformat(raw["departure"]),
            arrival=datetime.fromisoformat(raw["arrival"]),
            price_ore=int(raw["price_ore"]),
        )


class FlightProvider(Protocol):
    """Anything that can list priced direct flights for an airport pair on a date."""

    async def search(
        self, origin: Airport, destination: Airport, day: date
    ) -> list[FlightOffer]: ...


def _to_datetime(simple, day: date) -> datetime | None:
    """`fast_flights.model.SimpleDatetime` -> aware datetime.

    Its `date` and `time` are plain sequences, and the year is occasionally absent from
    Google's markup, in which case the searched day is the only sane assumption.
    """
    try:
        parts = list(simple.date)
        hour, minute = list(simple.time)[:2]
    except (TypeError, ValueError, AttributeError):
        return None
    if len(parts) == 3 and all(isinstance(p, int) for p in parts):
        year, month, dom = parts
    else:
        year, month, dom = day.year, day.month, day.day
    try:
        return datetime(year, month, dom, hour, minute, tzinfo=TZ)
    except ValueError:
        return None


class GoogleFlightsProvider:
    """Reads Google Flights via `fast-flights`, in a worker thread.

    `get_flights` is synchronous and does blocking network I/O, so calling it from the event
    loop would stall every other adapter running concurrently.
    """

    name = "google-flights"

    def __init__(self, *, currency: str = "SEK", consent_cookie: dict | None = None) -> None:
        self.currency = currency
        self.consent_cookie = consent_cookie or GOOGLE_CONSENT_COOKIE

    def _fetch_sync(self, origin: Airport, destination: Airport, day: date) -> list[FlightOffer]:
        try:
            from fast_flights import FlightQuery, Passengers, create_query  # noqa: PLC0415
            from fast_flights.parser import parse  # noqa: PLC0415
            from primp import Client  # noqa: PLC0415
        except ImportError as exc:
            # This one really is optional, so name the extra rather than leaving the caller
            # with a bare module name to guess from.
            raise RuntimeError(
                "flight pricing needs the `flights` extra: pip install -e '.[flights]'"
            ) from exc

        query = create_query(
            flights=[
                FlightQuery(
                    date=day.isoformat(),
                    from_airport=origin.iata,
                    to_airport=destination.iata,
                )
            ],
            trip="one-way",
            seat="economy",
            passengers=Passengers(adults=1),
            currency=self.currency,
            max_stops=0,
        )

        client = Client(
            impersonate="chrome_145", impersonate_os="macos", referer=True, cookie_store=True
        )
        client.set_cookies("https://www.google.com", self.consent_cookie)
        response = client.get("https://www.google.com/travel/flights", params=query.params())
        html = response.text
        if "consent.google.com" in html[:2000]:
            raise RuntimeError("Google returned its consent page; the consent cookie was rejected")

        try:
            results = parse(html)
        except (IndexError, AttributeError, ValueError, TypeError, KeyError) as exc:
            # `fast-flights` indexes into the parsed markup without checking. An airport pair
            # with no service at all - Malmo to Bromma, since Bromma's network shrank - yields
            # a payload whose flight slot is null, and the library raises whatever the missing
            # index happens to be (TypeError on payload[3][0], IndexError elsewhere). None of
            # these mean "broken scraper"; they mean "no flights", so they return an empty list.
            log.info("no parseable flights for %s-%s: %s", origin.iata, destination.iata, exc)
            return []

        return self._map(results, origin, destination, day)

    def _map(self, results, origin: Airport, destination: Airport, day: date) -> list[FlightOffer]:
        offers: list[FlightOffer] = []
        for result in results:
            legs = getattr(result, "flights", None) or []
            if len(legs) != 1:
                continue  # a connection, and this planner routes strictly within Sweden
            leg = legs[0]
            from_code = getattr(leg.from_airport, "code", "") or origin.iata
            to_code = getattr(leg.to_airport, "code", "") or destination.iata
            if not (is_swedish(from_code) and is_swedish(to_code)):
                continue
            departure = _to_datetime(leg.departure, day)
            arrival = _to_datetime(leg.arrival, day)
            price = getattr(result, "price", None)
            if departure is None or arrival is None or not isinstance(price, int) or price <= 0:
                continue
            if arrival < departure:
                # Google renders a next-day arrival without advancing the date field.
                arrival += timedelta(days=1)
            carriers = getattr(result, "airlines", None) or []
            offers.append(
                FlightOffer(
                    carrier=carriers[0] if carriers else "unknown",
                    from_iata=from_code.upper(),
                    to_iata=to_code.upper(),
                    departure=departure,
                    arrival=arrival,
                    price_ore=price * SEK,
                )
            )
        offers.sort(key=lambda o: (o.departure, o.price_ore))
        return offers

    async def search(
        self, origin: Airport, destination: Airport, day: date
    ) -> list[FlightOffer]:
        return await asyncio.to_thread(self._fetch_sync, origin, destination, day)


class NullFlightProvider:
    """Used when flight search is disabled or `fast-flights` is not installed."""

    name = "none"

    async def search(self, origin: Airport, destination: Airport, day: date) -> list[FlightOffer]:
        return []


def flight_route_addition(
    offers: list[FlightOffer],
    service_date: date,
    origin_stop,
    destination_stop,
) -> RouteAddition | None:
    """One synthetic route per airport pair, one trip per departure.

    Trips carry the exact fare, so McRAPTOR's price criterion needs no bound for them. The
    trips share a stop sequence and are sorted by departure, which is all RAPTOR requires.
    """
    if not offers:
        return None
    pair = (offers[0].from_iata, offers[0].to_iata)
    trips: list[Trip] = []
    for offer in offers:
        if (offer.from_iata, offer.to_iata) != pair:
            continue
        departure = to_service_seconds(offer.departure, service_date)
        arrival = to_service_seconds(offer.arrival, service_date)
        if arrival <= departure:
            continue
        trips.append(
            Trip(
                id=f"{offer.route_id}",
                arrivals=[departure, arrival],
                departures=[departure, arrival],
                headsign=offer.carrier,
                precomputed_fare_ore=offer.price_ore,
            )
        )
    if not trips:
        return None

    return RouteAddition(
        info=RouteInfo(
            # One RAPTOR route per airport pair, regardless of carrier: the route is the
            # stop sequence. Which airline flies each departure lives on the trip.
            id=f"air:{pair[0]}-{pair[1]}",
            mode=TransportMode.FLIGHT,
            operator=FLIGHT_OPERATOR_PREFIX,
            synthetic=True,
        ),
        stops=[origin_stop, destination_stop],
        trips=trips,
    )


def cache_key(origin: Airport, destination: Airport, day: date) -> str:
    return f"flights:{origin.iata}:{destination.iata}:{day.isoformat()}"
