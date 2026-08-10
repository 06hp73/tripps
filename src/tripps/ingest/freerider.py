"""Hertz Freerider inventory: fetch, parse, and cost.

Source: `GET https://www.hertzfreerider.se/api/transport-routes/?country=SWEDEN`.
This is the Gatsby front-end's own XHR, unauthenticated, plain JSON. There is no
official API and no documented schema, so `parse_offers` is defensive and the response
shape is pinned by a contract test against a recorded fixture.

Field semantics, established empirically from a live snapshot (2026-07-10, 79 routes)
rather than from documentation, which does not exist:

* `originalDistance` / `originalTravelTime` - the direct route: km and minutes.
* `distance` / `travelTime` - the *allowance*. Both are exactly the "original" value
  inflated by ~20% (measured ratio 1.20-1.21 across all 79 routes, never below 1.20).
  So `distance` is the included mileage, not the distance you drive.
* `availableAt` / `latestReturn` - the hard rental window. Naive ISO timestamps in
  Europe/Stockholm.
* `expireTime` - when the *offer* stops being bookable, independent of the window.

Opening hours are deliberately NOT treated as a feasibility constraint. In the live
snapshot 43 of 73 kiosk `latestReturn` deadlines fall outside the return station's
published hours, as does one staffed pickup. Those deadlines are set by Hertz itself, so
they cannot be infeasible; the published hours describe the staffed desk, not access to
the car. Enforcing them would silently discard the majority of real offers. They are
surfaced as an advisory warning instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from ..models import SEK, STOCKHOLM_TZ, Stop, TransportMode

log = logging.getLogger(__name__)

TZ = ZoneInfo(STOCKHOLM_TZ)
FREERIDER_OPERATOR = "hertz-freerider"

#: Ratio of `distance` to `originalDistance` observed across every live route.
#: A snapshot that violates this means the field semantics changed under us.
EXPECTED_ALLOWANCE_RATIO = 1.20
ALLOWANCE_RATIO_TOLERANCE = 0.05


def _parse_local(ts: str | None) -> datetime | None:
    """Freerider timestamps are naive local time. Attach Europe/Stockholm explicitly."""
    if not ts:
        return None
    return datetime.fromisoformat(ts).replace(tzinfo=TZ)


@dataclass(frozen=True, slots=True)
class OpeningHours:
    """Per ISO weekday (1=Monday .. 7=Sunday). None means closed that day."""

    by_weekday: dict[int, tuple[str, str] | None]

    def is_open_at(self, when: datetime) -> bool:
        window = self.by_weekday.get(when.isoweekday())
        if window is None:
            return False
        open_s, close_s = window
        t = when.time()
        return (
            datetime.fromisoformat(open_s).time()
            <= t
            <= datetime.fromisoformat(close_s).time()
        )


@dataclass(frozen=True, slots=True)
class FreeriderStation:
    trac_code: str
    name: str
    city: str
    lat: float
    lon: float
    kiosk: bool
    #: ISO-3166 alpha-2, lowercase, as the endpoint reports it ("se", "no").
    country: str = "se"
    address: str = ""
    hours: OpeningHours | None = None

    @property
    def stop_id(self) -> str:
        return f"freerider:{self.trac_code}"

    def to_stop(self) -> Stop:
        return Stop(
            id=self.stop_id,
            name=f"{self.name}",
            lat=self.lat,
            lon=self.lon,
            modes=frozenset({TransportMode.FREERIDER}),
            external_ids={"freerider_trac": self.trac_code},
        )


@dataclass(frozen=True, slots=True)
class FreeriderOffer:
    route_id: int
    offer_id: int
    pickup: FreeriderStation
    dropoff: FreeriderStation
    available_at: datetime
    latest_return: datetime
    expire_time: datetime | None
    car_model: str
    #: Included mileage (the allowance). Exceeding it triggers Hertz's excess-mileage fee.
    included_km: float
    #: The direct route you would actually drive.
    direct_km: float
    allowed_minutes: int
    direct_minutes: int

    @property
    def drive_seconds(self) -> int:
        return self.direct_minutes * 60

    def is_live(self, now: datetime) -> bool:
        """Bookable and not yet past its return deadline."""
        if self.expire_time is not None and now >= self.expire_time:
            return False
        return now < self.latest_return

    def latest_pickup(self) -> datetime:
        """Last moment you can collect the car and still return it in time."""
        return self.latest_return - timedelta(seconds=self.drive_seconds)

    def pickup_outside_staffed_hours(self, when: datetime) -> bool:
        return self.pickup.hours is not None and not self.pickup.hours.is_open_at(when)

    def allowance_ratio(self) -> float:
        return self.included_km / self.direct_km if self.direct_km else 0.0


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FuelParams:
    """Assumptions about how far the free first tank goes and what fuel costs after."""

    tank_range_km: float
    consumption_l_per_100km: float
    fuel_price_ore_per_litre: int

    def fuel_cost_ore(self, driven_km: float) -> int:
        beyond = max(0.0, driven_km - self.tank_range_km)
        litres = beyond / 100.0 * self.consumption_l_per_100km
        return int(litres * self.fuel_price_ore_per_litre)


#: Optimistic: a big tank, an efficient car, cheap fuel. Used for the ROUTING floor,
#: where the contract demands a value at or below the truth.
OPTIMISTIC_FUEL = FuelParams(
    tank_range_km=750.0, consumption_l_per_100km=5.0, fuel_price_ore_per_litre=1700
)

#: Typical: what a real traveller is likely to pay. Used for the phase-2 ESTIMATE.
TYPICAL_FUEL = FuelParams(
    tank_range_km=550.0, consumption_l_per_100km=6.5, fuel_price_ore_per_litre=1900
)


@dataclass(frozen=True, slots=True)
class FreeriderCostModel:
    """What a Freerider car actually costs the traveller.

    The car, the rental and the first tank are free; this is why Freerider legs make
    mixed-mode itineraries so cheap. The cost is therefore *not* a flat zero:

    * fuel burned beyond the first tank, on offers longer than a tank's range;
    * Hertz's excess-mileage ("övermil") fee, if you drive past `included_km`.

    Hertz publishes neither the tank range per car class nor the excess-mileage rate in
    any machine-readable place - hertzfreerider.se is a Prismic-backed SPA with no terms
    endpoint. Rather than hardcode invented numbers and present them as fact, these are
    explicit, overridable assumptions, and every quote this model produces is emitted
    with `PriceConfidence.ESTIMATED` and a note naming the assumption.

    Two parameter sets, because the two consumers want opposite errors:
    `floor_ore` must never exceed the truth (it prunes the search), while
    `estimate_ore` should be close to it (it is shown to a human).
    """

    optimistic: FuelParams = OPTIMISTIC_FUEL
    typical: FuelParams = TYPICAL_FUEL
    #: Excess-mileage fee beyond `included_km`. Unverified; only bites on detours.
    overmil_ore_per_km: int = 3 * SEK

    def _overmil_ore(self, offer: FreeriderOffer, driven_km: float) -> int:
        excess = max(0.0, driven_km - offer.included_km)
        return int(excess * self.overmil_ore_per_km)

    def floor_ore(self, offer: FreeriderOffer, driven_km: float | None = None) -> int:
        """Lower bound on cost, safe to use as a McRAPTOR criterion.

        Assumes the traveller drives the direct route (so no excess mileage) in the most
        favourable car at the cheapest fuel. Never above the true cost.
        """
        driven = offer.direct_km if driven_km is None else driven_km
        return self.optimistic.fuel_cost_ore(driven)

    def estimate_ore(self, offer: FreeriderOffer, driven_km: float | None = None) -> int:
        """Best guess at what the traveller pays. Shown to humans, flagged ESTIMATED."""
        driven = offer.direct_km if driven_km is None else driven_km
        return self.typical.fuel_cost_ore(driven) + self._overmil_ore(offer, driven)

    def explain(self, offer: FreeriderOffer) -> str:
        estimate = self.estimate_ore(offer)
        if estimate == 0:
            return (
                f"Car, rental and first tank are free; {offer.direct_km:.0f} km fits "
                f"within one tank and the {offer.included_km:.0f} km mileage allowance."
            )
        beyond = max(0.0, offer.direct_km - self.typical.tank_range_km)
        return (
            f"Free car and first tank, but {offer.direct_km:.0f} km exceeds an assumed "
            f"{self.typical.tank_range_km:.0f} km tank range by {beyond:.0f} km; "
            f"estimated fuel {estimate / SEK:.0f} SEK. Assumption, not a Hertz quote."
        )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _parse_hours(raw: dict | None) -> OpeningHours | None:
    if not raw:
        return None
    by_weekday: dict[int, tuple[str, str] | None] = {}
    for key, value in raw.items():
        try:
            day = int(key)
        except (TypeError, ValueError):
            continue
        if not value or not value.get("openingTime") or not value.get("closingTime"):
            by_weekday[day] = None
        else:
            by_weekday[day] = (value["openingTime"], value["closingTime"])
    return OpeningHours(by_weekday=by_weekday)


def _parse_station(raw: dict) -> FreeriderStation:
    return FreeriderStation(
        trac_code=raw["tracCode"],
        name=raw.get("name") or raw["tracCode"],
        city=raw.get("city") or "",
        lat=float(raw["geoLat"]),
        lon=float(raw["geoLon"]),
        kiosk=bool(raw.get("kiosk", False)),
        country=(raw.get("country") or "se").lower(),
        address=raw.get("address") or "",
        hours=_parse_hours(raw.get("regularOpeningHours")),
    )


def parse_offers(payload: list[dict]) -> list[FreeriderOffer]:
    """Flatten the grouped response into offers, skipping anything malformed.

    The endpoint groups routes by (pickupLocationName, returnLocationName); the group
    level carries no information the route level lacks, so it is discarded.
    """
    offers: list[FreeriderOffer] = []
    for group in payload:
        for raw in group.get("routes", []):
            try:
                available_at = _parse_local(raw["availableAt"])
                latest_return = _parse_local(raw["latestReturn"])
                if available_at is None or latest_return is None:
                    raise KeyError("availableAt/latestReturn")
                offer = FreeriderOffer(
                    route_id=int(raw["id"]),
                    offer_id=int(raw["transportOfferId"]),
                    pickup=_parse_station(raw["pickupLocation"]),
                    dropoff=_parse_station(raw["returnLocation"]),
                    available_at=available_at,
                    latest_return=latest_return,
                    expire_time=_parse_local(raw.get("expireTime")),
                    car_model=raw.get("carModel") or "",
                    included_km=float(raw["distance"]),
                    direct_km=float(raw["originalDistance"]),
                    allowed_minutes=int(raw["travelTime"]),
                    direct_minutes=int(raw["originalTravelTime"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("skipping malformed freerider route: %s", exc)
                continue

            if offer.direct_km <= 0 or offer.direct_minutes <= 0:
                log.warning("skipping freerider route %s with non-positive distance", offer.route_id)
                continue
            if offer.latest_return <= offer.available_at:
                log.warning("skipping freerider route %s with empty window", offer.route_id)
                continue
            offers.append(offer)
    return offers


def only_within(offers: list[FreeriderOffer], country: str = "se") -> list[FreeriderOffer]:
    """Keep offers whose pickup *and* dropoff are in `country`.

    The endpoint's `?country=SWEDEN` filter selects by destination, not by both endpoints:
    a live snapshot contains a Kirkenes (Norway) -> Stockholm repositioning. This planner
    routes strictly within Sweden, so cross-border offers are dropped rather than silently
    producing an itinerary that starts in another country.
    """
    country = country.lower()
    return [
        o
        for o in offers
        if o.pickup.country == country and o.dropoff.country == country
    ]


def crossing_into(offers: list[FreeriderOffer], country: str = "se") -> list[FreeriderOffer]:
    """Offers that end in `country` but start outside it.

    `only_within` drops these, which is right for routing — no Swedish timetable reaches
    Sandnessjøen, so an itinerary starting there could never be built. But it leaves our car
    count lower than the number on hertzfreerider.se, whose Sweden page lists them, and a
    number that quietly disagrees with the operator's own site invites doubt about every
    other number we print. So we count them and say so, rather than hiding the difference.
    """
    country = country.lower()
    return [
        o
        for o in offers
        if o.dropoff.country == country and o.pickup.country != country
    ]


def schema_drift(offers: list[FreeriderOffer]) -> list[str]:
    """Contract check: has the meaning of `distance` changed?

    `distance` being ~1.2x `originalDistance` is the only evidence we have that it means
    "included mileage". If a future snapshot breaks that, the cost model is silently
    wrong and must not be trusted.
    """
    problems: list[str] = []
    for offer in offers:
        ratio = offer.allowance_ratio()
        if abs(ratio - EXPECTED_ALLOWANCE_RATIO) > ALLOWANCE_RATIO_TOLERANCE:
            problems.append(
                f"offer {offer.route_id}: distance/originalDistance = {ratio:.3f}, "
                f"expected ~{EXPECTED_ALLOWANCE_RATIO}"
            )
    return problems


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class FreeriderClient:
    """Polls the inventory endpoint. Never raises upstream failures at the caller."""

    def __init__(
        self,
        base_url: str = "https://www.hertzfreerider.se",
        client: httpx.AsyncClient | None = None,
        user_agent: str = "tripps/0.1",
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._user_agent = user_agent
        self._timeout = timeout

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, headers={"User-Agent": self._user_agent}
            )
        return self._client

    async def fetch_raw(self, country: str = "SWEDEN") -> list[dict]:
        client = await self._http()
        resp = await client.get(
            f"{self.base_url}/api/transport-routes/",
            params={"country": country},
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON list, got {type(payload).__name__}")
        return payload

    async def fetch_offers(self, country: str = "SWEDEN") -> list[FreeriderOffer]:
        return parse_offers(await self.fetch_raw(country))

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def offers_to_log_rows(offers: list[FreeriderOffer], raw: list[dict] | None = None) -> list[dict]:
    """Shape offers for `Database.log_freerider_offers`."""
    raw_by_route: dict[int, dict] = {}
    for group in raw or []:
        for route in group.get("routes", []):
            try:
                raw_by_route[int(route["id"])] = route
            except (KeyError, TypeError, ValueError):
                continue
    return [
        {
            "offer_id": o.offer_id,
            "route_id": o.route_id,
            "pickup_trac": o.pickup.trac_code,
            "return_trac": o.dropoff.trac_code,
            "available_at": o.available_at.isoformat(),
            "latest_return": o.latest_return.isoformat(),
            "expire_time": o.expire_time.isoformat() if o.expire_time else None,
            "car_model": o.car_model,
            "distance_km": o.direct_km,
            "travel_minutes": o.direct_minutes,
            "payload": raw_by_route.get(o.route_id, {}),
        }
        for o in offers
    ]
