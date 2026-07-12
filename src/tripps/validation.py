"""Live-route validation: run canonical corridors through the real planner, check invariants.

Green unit tests prove the logic in isolation; this proves the planner is actually correct on
real routes and surfaces which live price sources are up. Meant for CI/cron: it exits non-zero
on a *hard* failure - a corridor that stops routing, or a price-floor violation (which means the
router could have pruned the genuinely cheapest journey before pricing). Degraded pricing
(a live source down in this environment) is a WARN, not a failure. A run also warms
`reprice_delta` rows, which feed floor calibration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

from .models import SearchConstraints, TransportMode
from .search import Planner

_TRAIN = TransportMode.TRAIN
_BUS = TransportMode.BUS
_FERRY = TransportMode.FERRY
_FREERIDER = TransportMode.FREERIDER
_FLIGHT = TransportMode.FLIGHT
_CONNECTIVE = {TransportMode.WALK, TransportMode.LOCAL_TRANSIT}


@dataclass(frozen=True, slots=True)
class Corridor:
    name: str
    origin: str
    destination: str
    modes: frozenset[TransportMode]
    note: str = ""


#: Canonical routes covering each mode + a couple of card/fare paths. Chosen so a break in
#: routing, pricing, or the new Flygbussarna fare table shows up as a WARN/FAIL here.
CORRIDORS: tuple[Corridor, ...] = (
    Corridor("Stockholm->Goteborg", "Stockholm C", "Göteborg C", frozenset({_TRAIN, _BUS, _FLIGHT}), "flagship intercity"),
    Corridor("Goteborg->Malmo", "Göteborg C", "Malmö C", frozenset({_TRAIN, _BUS}), "Öresundståg regional corridor"),
    Corridor("Malmo->Umea", "Malmö C", "Umeå C", frozenset({_TRAIN, _BUS, _FLIGHT}), "long, many transfers"),
    Corridor("Stockholm->Arlanda", "Stockholm C", "Arlanda", frozenset({_TRAIN, _BUS}), "airport coach fare + passage fee"),
    Corridor("Uppsala->Gavle", "Uppsala C", "Gävle C", frozenset({_TRAIN, _BUS}), "cross-border UL corridor"),
    Corridor("Stockholm->Sundsvall", "Stockholm C", "Sundsvall C", frozenset({_TRAIN, _BUS, _FREERIDER}), "mixed-mode / Freerider"),
)

PlannerFactory = Callable[[date], Planner]


@dataclass(slots=True)
class CorridorResult:
    corridor: Corridor
    status: str  # "pass" | "warn" | "fail"
    itineraries: int
    priced: int
    cheapest_sek: float | None
    modes: list[str]
    sources: dict[str, str]
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "fail"


def _has_floor_violation(warnings: Sequence[str]) -> bool:
    return any("floor violation" in w.lower() or "floor-violation" in w.lower() for w in warnings)


async def validate_one(
    make_planner: PlannerFactory, corridor: Corridor, service_date: date, offers
) -> CorridorResult:
    """Run one corridor and classify it against the invariants.

    fail = search errored, no itinerary on a route we can route, or a price-floor violation.
    warn = routed but nothing fully priced (live source down), or a suspicious 0-price total.
    pass = routed, priced, no floor violation.
    """
    constraints = SearchConstraints(
        allowed_modes=set(corridor.modes) | _CONNECTIVE,
        include_freerider=_FREERIDER in corridor.modes,
    )
    try:
        planner = make_planner(service_date)
        response, _stats = await planner.search(
            corridor.origin, corridor.destination, service_date,
            constraints=constraints, offers=offers,
        )
    except Exception as exc:  # noqa: BLE001 - any break in routing/pricing is a hard failure here
        return CorridorResult(
            corridor, "fail", 0, 0, None, [], {},
            [f"search error: {type(exc).__name__}: {exc}"],
        )

    itins = response.itineraries
    priced = [i for i in itins if i.total_price_ore is not None]
    modes = sorted({m.value for i in itins for m in i.modes if m.value != "walk"})
    cheapest = min((i.total_price_ore for i in priced), default=None)

    status = "pass"
    messages: list[str] = []
    if not itins:
        status = "fail"
        messages.append("no itinerary found on a routable corridor")
    if _has_floor_violation(response.warnings):
        status = "fail"
        messages.append("price-floor violation: the true cheapest journey may have been pruned")
    if itins and not priced:
        status = "fail" if status == "fail" else "warn"
        messages.append("routed but nothing fully priced (a live price source may be down)")
    if cheapest is not None and cheapest <= 0:
        status = "fail" if status == "fail" else "warn"
        messages.append("cheapest priced total is 0 with no card held - suspicious")

    return CorridorResult(
        corridor, status, len(itins), len(priced),
        (cheapest / 100 if cheapest is not None else None),
        modes, dict(response.source_status), messages,
    )


async def run_validation(
    make_planner: PlannerFactory,
    *,
    service_date: date,
    offers=None,
    corridors: Sequence[Corridor] = CORRIDORS,
) -> list[CorridorResult]:
    """Run every corridor sequentially (they share one orchestrator + rate limiters)."""
    results: list[CorridorResult] = []
    for corridor in corridors:
        results.append(await validate_one(make_planner, corridor, service_date, offers))
    return results


_MARK = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}


def render(results: Sequence[CorridorResult]) -> list[str]:
    lines: list[str] = []
    for r in results:
        price = f"{r.cheapest_sek:.0f} SEK" if r.cheapest_sek is not None else "unpriced"
        lines.append(
            f"{_MARK[r.status]} {r.corridor.name:22} "
            f"{r.itineraries:2d} itin, {r.priced} priced, cheapest {price:>10}  "
            f"modes={'+'.join(r.modes) or '-'}"
        )
        for m in r.messages:
            lines.append(f"         - {m}")
    return lines


def summarize(results: Sequence[CorridorResult]) -> tuple[int, int, int]:
    """(passed, warned, failed) counts."""
    failed = sum(1 for r in results if r.status == "fail")
    warned = sum(1 for r in results if r.status == "warn")
    return len(results) - failed - warned, warned, failed
