"""Helpers for building small, exactly-known timetables in tests."""

from __future__ import annotations

from tripps.models import Stop, TransportMode
from tripps.routing.timetable import RouteInfo, Timetable, TimetableBuilder, Trip

# Roughly-real coordinates so segment distances (and therefore price floors) are sane.
COORDS: dict[str, tuple[float, float]] = {
    "STO": (59.3300, 18.0590),  # Stockholm C
    "NRK": (58.5960, 16.1830),  # Norrkoping
    "LIN": (58.4160, 15.6250),  # Linkoping
    "GBG": (57.7089, 11.9746),  # Goteborg
    "MMX": (55.6090, 13.0000),  # Malmo
    "BLE": (60.4845, 15.4379),  # Borlange
    "ARN": (59.6519, 17.9186),  # Arlanda
}


def hhmm(hours: int, minutes: int = 0) -> int:
    """Service-day seconds. Values >= 24h are legal (GTFS after-midnight departures)."""
    return hours * 3600 + minutes * 60


def stop(code: str, modes: frozenset[TransportMode] = frozenset()) -> Stop:
    lat, lon = COORDS[code]
    return Stop(id=code, name=code, lat=lat, lon=lon, modes=modes)


class Net:
    """Fluent builder: `Net().route(...).transfer(...).build()`."""

    def __init__(self) -> None:
        self.b = TimetableBuilder()

    def stops(self, *codes: str) -> Net:
        for code in codes:
            self.b.add_stop(stop(code))
        return self

    def route(
        self,
        route_id: str,
        codes: list[str],
        trips: list[list[tuple[int, int]]],
        *,
        mode: TransportMode = TransportMode.TRAIN,
        operator: str | None = "TEST",
        fares_ore: list[int | None] | None = None,
        synthetic: bool = False,
    ) -> Net:
        """`trips[t][i]` is the (arrival, departure) pair at `codes[i]` for trip t."""
        self.stops(*codes)
        info = RouteInfo(id=route_id, mode=mode, operator=operator, synthetic=synthetic)
        fares = fares_ore or [None] * len(trips)
        for t, times in enumerate(trips):
            self.b.add_trip(
                info,
                codes,
                Trip(
                    id=f"{route_id}#{t}",
                    arrivals=[a for a, _ in times],
                    departures=[d for _, d in times],
                    precomputed_fare_ore=fares[t],
                ),
            )
        return self

    def transfer(self, a: str, b: str, seconds: int) -> Net:
        self.b.add_transfer(a, b, seconds)
        return self

    def build(self) -> Timetable:
        return self.b.build()


def at(t: int) -> tuple[int, int]:
    """A stop where arrival == departure (no dwell)."""
    return (t, t)
