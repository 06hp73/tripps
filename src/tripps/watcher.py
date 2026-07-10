"""Freerider watches: standing interest in a free-car route, matched as inventory arrives.

Free Hertz cars are the headline feature, but they are first-come and vanish in minutes, so
you only ever find one if it happens to exist for the exact date you searched. A watch flips
that: register a route once, and the background poller that already refreshes the inventory
every few minutes matches new cars against it and announces the first time one appears - the
free car finds you.

Matching is geographic, not by station id, because Freerider stations sit at dealerships and
airports outside the city (Arlanda is 40 km from Stockholm C), so a watch stores the
origin/destination coordinates and a radius. A car matches when its pickup is within the
radius of the origin *and* its dropoff within the radius of the destination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .ingest.freerider import FreeriderOffer
from .routing.timetable import haversine_km

log = logging.getLogger(__name__)

#: Default match radius. Generous because Freerider depots are often well outside town, but
#: not so wide that a car to a different city counts.
DEFAULT_WATCH_RADIUS_KM = 40.0


@dataclass(frozen=True, slots=True)
class Watch:
    id: int
    origin: str
    destination: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    radius_km: float
    webhook_url: str | None = None

    @classmethod
    def from_row(cls, row) -> Watch:
        return cls(
            id=row["id"],
            origin=row["origin"],
            destination=row["destination"],
            origin_lat=row["origin_lat"],
            origin_lon=row["origin_lon"],
            dest_lat=row["dest_lat"],
            dest_lon=row["dest_lon"],
            radius_km=row["radius_km"],
            webhook_url=row["webhook_url"],
        )

    def matches(self, offer: FreeriderOffer) -> bool:
        return (
            haversine_km(self.origin_lat, self.origin_lon, offer.pickup.lat, offer.pickup.lon)
            <= self.radius_km
            and haversine_km(self.dest_lat, self.dest_lon, offer.dropoff.lat, offer.dropoff.lon)
            <= self.radius_km
        )


def match_watches(
    watches: list[Watch], offers: list[FreeriderOffer]
) -> list[tuple[Watch, FreeriderOffer]]:
    """Every (watch, car) pair where the car is on the watched route."""
    return [(w, o) for w in watches for o in offers if w.matches(o)]


def hit_payload(watch: Watch, offer: FreeriderOffer) -> dict:
    """The JSON body posted to a watch's webhook, and shown in the CLI/API."""
    return {
        "watch_id": watch.id,
        "route": f"{watch.origin} -> {watch.destination}",
        "pickup": offer.pickup.name,
        "dropoff": offer.dropoff.name,
        "available_at": offer.available_at.isoformat(),
        "latest_return": offer.latest_return.isoformat(),
        "car": offer.car_model,
        "direct_km": offer.direct_km,
        "book_url": "https://www.hertzfreerider.se/sv-se/",
    }


async def notify_webhook(url: str, payload: dict, *, timeout: float = 10.0) -> bool:
    """POST a hit to a watch's webhook. Never raises; returns whether it was delivered."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            return resp.status_code < 400
    except httpx.HTTPError as exc:
        log.warning("watch webhook %s failed: %s", url, exc)
        return False


def upcoming_cars_on_route(
    offers: list[FreeriderOffer],
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    *,
    radius_km: float = DEFAULT_WATCH_RADIUS_KM,
) -> list[FreeriderOffer]:
    """Free cars on this origin/destination route, any date, soonest available first.

    Used to surface the killer feature inside a normal search: a user looking for Stockholm
    -> Göteborg on the 5th is told there is a free car on the route on the 12th, which a
    date-specific search would never reveal.
    """
    probe = Watch(
        id=0,
        origin="",
        destination="",
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        radius_km=radius_km,
    )
    matched = [o for o in offers if probe.matches(o)]
    matched.sort(key=lambda o: o.available_at)
    return matched
