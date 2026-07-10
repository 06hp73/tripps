"""Swedish commercial airports.

Domestic flights are absent from every Trafiklab dataset - Samtrafiken's own ResRobot page
states the included modes are train, bus, tram, metro and ferry, and the `products=1`
"air traffic" bit is dead legacy from the underlying Hafas platform. So the flight layer
carries its own stop table.

The registry is generated from OurAirports (public domain), filtered to Swedish airports
with an IATA code and scheduled service. Coordinates come from that dataset rather than
from memory, because a wrong airport coordinate silently corrupts every distance, every
walk link and every price floor that depends on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..models import Stop, TransportMode
from ..routing.timetable import Timetable, haversine_km

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "airports.json"

#: How close a timetabled stop must be to count as "the airport's station".
#: Arlanda Express and Flygbussarna call at stops inside the airport, well within 3 km.
AIRPORT_STOP_RADIUS_KM = 3.0


@dataclass(frozen=True, slots=True)
class Airport:
    iata: str
    name: str
    lat: float
    lon: float
    municipality: str

    @property
    def stop_id(self) -> str:
        return f"air:{self.iata}"

    def to_stop(self) -> Stop:
        return Stop(
            id=self.stop_id,
            name=self.name,
            lat=self.lat,
            lon=self.lon,
            modes=frozenset({TransportMode.FLIGHT}),
            external_ids={"iata": self.iata},
        )


@lru_cache(maxsize=1)
def load_airports() -> dict[str, Airport]:
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {
        entry["iata"]: Airport(
            iata=entry["iata"],
            name=entry["name"],
            lat=entry["lat"],
            lon=entry["lon"],
            municipality=entry.get("municipality", ""),
        )
        for entry in raw
    }


def is_swedish(iata: str) -> bool:
    """Guards the "strictly within Sweden" requirement.

    A Google Flights search for Stockholm to Goteborg happily returns itineraries
    connecting through Frankfurt and Helsinki. Those are not domestic trips.
    """
    return iata.upper() in load_airports()


def nearest_airports(
    lat: float, lon: float, *, limit: int = 2, max_km: float = 120.0
) -> list[Airport]:
    """Airports plausibly serving a place, nearest first.

    The radius is generous because Swedish airports sit far from their cities (Skavsta is
    100 km from Stockholm), and narrow enough that Kiruna never serves Malmo.
    """
    scored = [
        (haversine_km(lat, lon, airport.lat, airport.lon), airport)
        for airport in load_airports().values()
    ]
    scored = [pair for pair in scored if pair[0] <= max_km]
    scored.sort(key=lambda pair: pair[0])
    return [airport for _km, airport in scored[:limit]]


def resolve_airport_stop(timetable: Timetable, airport: Airport) -> Stop | None:
    """The timetabled stop that *is* this airport, if the feed has one.

    Reusing the feed's own airport stop is what makes a flight leg reachable: the airport
    coach and the airport train already call there. Inventing a separate stop and walking
    to it would either strand the flight or pretend you can walk 40 km to Arlanda.
    """
    candidates: list[tuple[float, Stop]] = []
    for stop in timetable.stops:
        km = haversine_km(airport.lat, airport.lon, stop.lat, stop.lon)
        if km <= AIRPORT_STOP_RADIUS_KM:
            candidates.append((km, stop))
    if not candidates:
        return None

    def score(pair: tuple[float, Stop]) -> tuple[int, float]:
        km, stop = pair
        name = stop.name.casefold()
        named = 0 if ("flygplats" in name or "airport" in name) else 1
        return (named, km)

    candidates.sort(key=score)
    return candidates[0][1]
