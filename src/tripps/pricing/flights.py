"""Prices a domestic flight leg from the offers already fetched for this search.

Flight offers arrive priced: the search that discovered the departure also returned its
fare, and that fare is what the synthetic trip carries. Re-scraping Google Flights once per
leg during phase 2 would double every request for information already in hand, so this
adapter reads back the offer it was given.

The quote is EXACT because it came from a live sales page for that departure, but it is a
*scraped* exactness: Google can change its markup, and the itinerary is bookable only on
the airline's own site. The deeplink says where.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from ..ingest.flights import FlightOffer
from ..models import Leg, Quote, TransportMode
from .base import HttpPriceAdapter, unavailable
from .base import exact as exact_quote

SOURCE = "google-flights"


def deeplink(offer: FlightOffer) -> str:
    query = quote_plus(
        f"Flights from {offer.from_iata} to {offer.to_iata} "
        f"on {offer.departure:%Y-%m-%d}"
    )
    return f"https://www.google.com/travel/flights?q={query}"


class FlightAdapter(HttpPriceAdapter):
    """Serves prices from the offer snapshot the network was built with."""

    name = SOURCE
    modes = frozenset({TransportMode.FLIGHT})

    def __init__(self, offers: list[FlightOffer] | None = None) -> None:
        super().__init__(min_interval=0.0)
        self.offers: dict[str, FlightOffer] = {}
        if offers:
            self.load(offers)

    def load(self, offers: list[FlightOffer]) -> None:
        self.offers = {offer.route_id: offer for offer in offers}

    def supports(self, leg: Leg) -> bool:
        return leg.mode is TransportMode.FLIGHT

    def _offer_for(self, leg: Leg) -> FlightOffer | None:
        if leg.service_ref and leg.service_ref in self.offers:
            return self.offers[leg.service_ref]
        # Fall back to matching the departure, in case the trip id was rewritten.
        for offer in self.offers.values():
            if offer.departure == leg.departure and offer.to_iata in leg.to_stop.name.upper():
                return offer
        return None

    async def quote_leg(self, leg: Leg) -> Quote:
        if not self.supports(leg):
            return unavailable(self.name, None, "not a flight leg")

        offer = self._offer_for(leg)
        if offer is None:
            return unavailable(
                self.name,
                "https://www.google.com/travel/flights",
                "the flight offer for this leg is no longer held in memory",
            )

        return exact_quote(
            self.name,
            offer.price_ore,
            deeplink=deeplink(offer),
            fare_class="economy, one way",
            note=(
                f"{offer.carrier} {offer.from_iata}-{offer.to_iata}. "
                "Scraped from Google Flights; book on the airline's own site."
            ),
        )
