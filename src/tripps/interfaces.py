"""Stable seams between the routing core and the outside world.

Every unofficial endpoint this project depends on (SJ, FlixBus, Freerider, Google
Flights) is an undocumented internal that can change without notice. Five earlier
open-source Freerider scrapers died on a single site rewrite. So each source lives
behind `PriceAdapter`, is contract-tested against its real response schema, and can
fail without taking the search down: a dead adapter yields an UNAVAILABLE quote with
a deeplink, never an exception that empties the result list.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import Leg, Quote, Stop, TransportMode


class HealthState(str, enum.Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
    DISABLED = "disabled"  # missing credential, or switched off by config


@dataclass(frozen=True)
class SourceHealth:
    name: str
    state: HealthState
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.state in (HealthState.OK, HealthState.DEGRADED)


class PriceAdapter(ABC):
    """Turns one scheduled leg into a price.

    Implementations must not raise for upstream failures. They catch, and return
    `Quote.unavailable(...)`. Raising is reserved for programming errors.
    """

    #: Stable identifier used in caches, health maps and logs.
    name: str
    #: Modes this adapter can price at all.
    modes: frozenset[TransportMode]

    @abstractmethod
    def supports(self, leg: Leg) -> bool:
        """Can this adapter price this specific leg (right mode, right operator)?"""

    @abstractmethod
    async def quote_leg(self, leg: Leg) -> Quote:
        """Price one leg. Never raises on upstream failure."""

    async def quote_itinerary(self, legs: list[Leg]) -> Quote | None:
        """Optional through-fare for a run of consecutive legs on this source.

        Swedish Resplus through-tickets can price a multi-leg journey as one product,
        and that product is not always the sum of the legs. Adapters that can answer
        this override it; the orchestrator takes the cheaper of through vs sum.
        Returning None means "no through-fare concept here".
        """
        return None

    async def health(self) -> SourceHealth:
        return SourceHealth(self.name, HealthState.OK)

    async def aclose(self) -> None:
        """Release HTTP clients etc."""


class PriceFloor(ABC):
    """A guaranteed-not-too-high price estimate, used as a routing criterion.

    McRAPTOR needs a price on every leg *during* the search, but real prices are only
    knowable per concrete itinerary via a live lookup we cannot afford per label. So
    the router optimizes a lower bound and phase 2 resolves the truth.

    Correctness contract: `floor_ore` MUST NOT exceed the true price of that leg. An
    over-estimate can prune the genuinely cheapest itinerary before it is ever priced.
    An under-estimate only costs us a wider Pareto frontier.

    The bound is decomposed into a once-per-boarding component and a per-kilometre
    component so that it accumulates *additively* as the router rides along a route.
    Real fares are sub-additive in distance (a through ticket beats the sum of its
    segments), so an additive bound built from the minimum marginal per-km rate stays
    below the truth, which is exactly the direction the contract requires.
    """

    @abstractmethod
    def boarding_ore(self, mode: TransportMode, operator: str | None) -> int:
        """Lower bound on the cost of boarding at all, independent of distance."""

    @abstractmethod
    def per_km_ore(self, mode: TransportMode, operator: str | None) -> int:
        """Lower bound on the marginal cost of one more kilometre on this service."""

    def floor_ore(self, mode: TransportMode, operator: str | None, distance_km: float) -> int:
        return self.boarding_ore(mode, operator) + int(
            self.per_km_ore(mode, operator) * distance_km
        )


class StopResolver(ABC):
    """Free-text place name -> Stop."""

    @abstractmethod
    async def resolve(self, query: str, limit: int = 5) -> list[Stop]:
        """Best-effort candidates, best first. Empty list when nothing matches."""


class RoadMatrix(ABC):
    """Driving distance and time between two points, for Freerider legs.

    The Freerider payload carries its own `distance`/`travelTime`, but those describe
    Hertz's intended repositioning route, not necessarily the shortest path, and they
    are absent for the walk/transfer edges we synthesize. We keep an independent
    source and cross-check.
    """

    @abstractmethod
    async def route(self, origin: Stop, destination: Stop) -> tuple[float, int]:
        """Return (distance_km, duration_seconds)."""
