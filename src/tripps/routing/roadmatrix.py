"""Driving distance/time between two points, for Hertz Freerider legs.

Two backends:

* `OsrmRoadMatrix` queries a self-hosted OSRM (`/route/v1/driving/...`). Preferred.
* `HaversineRoadMatrix` needs no service at all and estimates road distance by inflating
  the great-circle distance with a detour factor. This is what the planner uses out of
  the box so a fresh clone runs with zero infrastructure.

The Freerider payload does carry its own `distance` and `travelTime`, but those describe
the route Hertz wants the car repositioned along, and the terms penalise driving well
beyond it. So the offer's own numbers are authoritative for the *contract* (see
`pricing.freerider`), while this matrix supplies an independent estimate used to sanity
check them and to cost the detour a traveller would actually make.
"""

from __future__ import annotations

import httpx

from ..interfaces import RoadMatrix
from ..models import Stop
from .timetable import haversine_km

#: Ratio of real Swedish road distance to great-circle distance. Empirically ~1.2-1.3
#: on the trunk network; 1.25 keeps the estimate close without pretending to precision.
DETOUR_FACTOR = 1.25

#: Average door-to-door speed including stops, on mixed motorway/rural roads.
AVERAGE_SPEED_KMH = 80.0


class HaversineRoadMatrix(RoadMatrix):
    """Infrastructure-free estimate. Good enough to rank itineraries, not to navigate."""

    def __init__(
        self, detour_factor: float = DETOUR_FACTOR, speed_kmh: float = AVERAGE_SPEED_KMH
    ) -> None:
        if detour_factor < 1.0:
            raise ValueError("detour factor cannot shorten the great-circle distance")
        if speed_kmh <= 0:
            raise ValueError("speed must be positive")
        self.detour_factor = detour_factor
        self.speed_kmh = speed_kmh

    async def route(self, origin: Stop, destination: Stop) -> tuple[float, int]:
        straight = haversine_km(origin.lat, origin.lon, destination.lat, destination.lon)
        distance_km = straight * self.detour_factor
        seconds = int(distance_km / self.speed_kmh * 3600)
        return distance_km, seconds


class OsrmRoadMatrix(RoadMatrix):
    """Self-hosted OSRM. Falls back to the estimate when the service is unreachable.

    A dead routing container must not empty the results list: a Freerider leg with an
    approximate duration is far more useful than no Freerider leg at all.
    """

    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        fallback: RoadMatrix | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self.fallback = fallback or HaversineRoadMatrix()

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def route(self, origin: Stop, destination: Stop) -> tuple[float, int]:
        coords = f"{origin.lon},{origin.lat};{destination.lon},{destination.lat}"
        url = f"{self.base_url}/route/v1/driving/{coords}"
        try:
            client = await self._http()
            resp = await client.get(url, params={"overview": "false", "alternatives": "false"})
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("code") != "Ok" or not payload.get("routes"):
                raise ValueError(f"osrm returned {payload.get('code')!r}")
            leg = payload["routes"][0]
            return leg["distance"] / 1000.0, int(leg["duration"])
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            return await self.fallback.route(origin, destination)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def build_road_matrix(osrm_base: str | None) -> RoadMatrix:
    return OsrmRoadMatrix(osrm_base) if osrm_base else HaversineRoadMatrix()
