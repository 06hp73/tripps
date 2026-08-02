"""Prices a Hertz Freerider leg.

Freerider is the only mode whose price needs no upstream lookup at all: the car, the
rental and the first tank are free by construction. The cost is what the traveller spends
beyond that, which the cost model computes from the offer's own distance.

Every quote is `ESTIMATED`, never `EXACT`, and says so. Hertz publishes neither the tank
range per car class nor the excess-mileage rate anywhere machine-readable -
hertzfreerider.se is a Prismic-backed SPA with no terms endpoint - so the numbers behind a
non-zero Freerider quote are our assumptions, not Hertz's tariff. Presenting them as exact
would be the single most misleading thing this planner could do, because it is precisely
the mode a user picks *because* it looks free.

Booking is out of scope: listing is public but reserving requires the user's own Azure AD
B2C session. The quote carries a deeplink and the itinerary carries a warning.
"""

from __future__ import annotations

from ..ingest.freerider import FreeriderCostModel, FreeriderOffer
from ..models import ADULT, Leg, Passenger, PassengerCategory, Quote, TransportMode
from .base import HttpPriceAdapter, estimated, unavailable

SOURCE = "freerider"
BOOKING_URL = "https://www.hertzfreerider.se/sv-se/"


def offer_key(offer: FreeriderOffer) -> str:
    return f"freerider:{offer.route_id}"


class FreeriderAdapter(HttpPriceAdapter):
    """Prices Freerider legs from the in-memory offer snapshot the router was built with.

    It makes no HTTP calls of its own: the inventory poller already fetched the offers, and
    re-fetching per leg would multiply requests against an undocumented endpoint for data
    we are already holding.
    """

    name = SOURCE
    modes = frozenset({TransportMode.FREERIDER})
    #: Every category, because none of them changes the number: a Freerider car costs its
    #: fuel whoever is driving. Claiming "no student fare here" would be noise, not honesty.
    sells_categories = frozenset(PassengerCategory)

    def __init__(
        self,
        offers: dict[str, FreeriderOffer] | None = None,
        cost_model: FreeriderCostModel | None = None,
    ) -> None:
        super().__init__(min_interval=0.0)
        self.offers = offers or {}
        self.cost_model = cost_model or FreeriderCostModel()

    def load(self, offers: list[FreeriderOffer]) -> None:
        self.offers = {offer_key(o): o for o in offers}

    def supports(self, leg: Leg) -> bool:
        return leg.mode is TransportMode.FREERIDER

    def _offer_for(self, leg: Leg) -> FreeriderOffer | None:
        if leg.service_ref:
            # Synthetic trip ids look like "freerider:<route_id>@<seconds>".
            base = leg.service_ref.split("@", 1)[0]
            offer = self.offers.get(base)
            if offer is not None:
                return offer
        for offer in self.offers.values():
            if (
                offer.pickup.stop_id == leg.from_stop.id
                and offer.dropoff.stop_id == leg.to_stop.id
            ):
                return offer
        return None

    async def quote_leg(self, leg: Leg, passenger: Passenger = ADULT) -> Quote:
        # The passenger category is deliberately ignored: the car is free whoever drives it,
        # and this amount is a fuel estimate, not a ticket that could carry a discount.
        if not self.supports(leg):
            return unavailable(self.name, BOOKING_URL, "not a Freerider leg")

        offer = self._offer_for(leg)
        if offer is None:
            return unavailable(
                self.name, BOOKING_URL, "the Freerider offer for this leg is no longer listed"
            )

        amount = self.cost_model.estimate_ore(offer)
        note = self.cost_model.explain(offer)
        if offer.expire_time is not None:
            note += f" Offer expires {offer.expire_time:%Y-%m-%d %H:%M}."
        return estimated(self.name, amount, deeplink=BOOKING_URL, note=note)

    def warnings_for(self, leg: Leg) -> list[str]:
        """Advisories a user needs before relying on a Freerider leg."""
        offer = self._offer_for(leg)
        if offer is None:
            return ["This Freerider car is no longer listed."]

        warnings = [
            "Freerider must be booked on hertzfreerider.se with a Hertz account; "
            "it cannot be reserved from here, and cars are taken quickly.",
            f"Return by {offer.latest_return:%Y-%m-%d %H:%M} at the latest. "
            f"Included mileage {offer.included_km:.0f} km.",
        ]
        if offer.pickup_outside_staffed_hours(leg.departure) and not offer.pickup.kiosk:
            warnings.append(
                f"Pickup at {offer.pickup.name} falls outside the station's staffed hours."
            )
        if self.cost_model.estimate_ore(offer) > 0:
            warnings.append(
                "Estimated fuel cost beyond the free first tank; the exact amount depends "
                "on the car and fuel prices, and is not quoted by Hertz."
            )
        return warnings
