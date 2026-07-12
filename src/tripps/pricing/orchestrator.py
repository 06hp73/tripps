"""Phase 2: replace routing price floors with real quotes, then re-rank.

Phase 1 (McRAPTOR) optimizes a *lower bound* on price, because real Swedish fares are
yield-managed and only knowable per concrete itinerary. That leaves a frontier of
candidates whose true order is still unknown. This module resolves it.

The shape is the one Entur uses: plan the journey on schedules, then price the surviving
patterns with a separate offer lookup, cache aggressively, and show the user what is
actually purchasable.

Three properties are load-bearing:

* **Bounded fan-out.** Pricing every candidate leg against every source would be dozens of
  requests to undocumented endpoints per search box submission. `CallBudget`, a per-source
  rate limiter, a candidate cap and a wall-clock deadline bound it.
* **Honest totals.** An itinerary with an unpriced vehicle leg has `total_price_ore ==
  None` and is ranked below every fully priced one, rather than being summed as if the
  missing leg were free. That failure mode would make broken pricing look like a bargain.
* **Detectable floor violations.** Every (floor, actual) pair is logged. If a floor ever
  exceeded the real price, McRAPTOR may have pruned the true cheapest journey. That is a
  correctness bug, and it is recorded rather than hidden.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..config import CacheTTL, PricingBudget
from ..db import Database
from ..interfaces import PriceAdapter
from ..models import (
    Itinerary,
    Leg,
    PriceConfidence,
    Quote,
    SearchConstraints,
    TransportMode,
)
from ..routing.floors import PriceFloorModel
from ..routing.journey import (
    collapse_equivalent,
    leg_distance_km,
    pattern_key,
    spread_by_departure,
)
from .base import BudgetExceeded, CallBudget
from .freerider import FreeriderAdapter

log = logging.getLogger(__name__)

#: Quote source used by the travel-card pass adapter; recognised here so a card-covered free
#: leg is not mistaken for a price-floor violation.
PASS_SOURCE = "travelcard"

#: Ordering of confidences when the UI must pick "the weakest link".
_CONFIDENCE_RANK = {
    PriceConfidence.EXACT: 0,
    PriceConfidence.CACHED: 1,
    PriceConfidence.STALE: 2,
    PriceConfidence.ESTIMATED: 3,
    PriceConfidence.UNAVAILABLE: 4,
}


@dataclass(slots=True)
class PricingResult:
    itineraries: list[Itinerary]
    warnings: list[str] = field(default_factory=list)
    source_status: dict[str, str] = field(default_factory=dict)
    calls_made: int = 0
    floor_violations: int = 0


@dataclass(slots=True)
class _Context:
    """Per-search state: the call budget and any floor violations observed."""

    call_budget: CallBudget
    violations: list[str] = field(default_factory=list)


class PricingOrchestrator:
    """Prices candidate itineraries concurrently, within a budget, and re-ranks them."""

    def __init__(
        self,
        adapters: list[PriceAdapter],
        db: Database | None = None,
        *,
        budget: PricingBudget | None = None,
        ttl: CacheTTL | None = None,
        floors: PriceFloorModel | None = None,
    ) -> None:
        self.adapters = adapters
        self.db = db
        self.budget = budget or PricingBudget()
        self.ttl = ttl or CacheTTL()
        self.floors = floors or PriceFloorModel()

    # --- adapter selection ------------------------------------------------

    def _adapter_for(self, leg: Leg) -> PriceAdapter | None:
        for adapter in self.adapters:
            if adapter.supports(leg):
                return adapter
        return None

    def _leg_priceable(self, leg: Leg) -> bool:
        """Can this leg get a real amount from some adapter (not just a booking link)?"""
        if leg.mode is TransportMode.WALK:
            return True
        adapter = self._adapter_for(leg)
        return adapter is not None and getattr(adapter, "provides_price", True)

    def _itinerary_priceable(self, itin: Itinerary) -> bool:
        return all(self._leg_priceable(leg) for leg in itin.legs)

    # --- one leg ----------------------------------------------------------

    async def _quote_leg(self, leg: Leg, ctx: _Context) -> Quote:
        """Cache -> budget -> upstream. Never raises."""
        adapter = self._adapter_for(leg)
        if adapter is None:
            return Quote.unavailable(
                source="none", note=f"no price source for a {leg.mode.value} leg"
            )

        ttl = self.ttl.for_mode(leg.mode)

        if self.db is not None:
            cached = self.db.get_quote(adapter.name, leg)
            if cached is not None and cached.confidence in (
                PriceConfidence.CACHED,
                PriceConfidence.ESTIMATED,
            ):
                self._check_floor(leg, cached, ctx)
                return cached

        try:
            quote = await adapter.quote_leg(leg)
        except BudgetExceeded as exc:
            # The source ran out of requests for this search. A stale cached number beats no
            # number, and it is labelled stale rather than passed off as current.
            if self.db is not None:
                stale = self.db.get_quote(adapter.name, leg)
                if stale is not None and stale.is_priced:
                    return stale
            return Quote.unavailable(source=adapter.name, note=str(exc))
        except Exception as exc:  # noqa: BLE001 - an adapter bug must not empty the results
            log.exception("adapter %s raised on leg %s", adapter.name, leg.service_ref)
            return Quote.unavailable(source=adapter.name, note=f"pricing failed: {exc}")

        if self.db is not None:
            self.db.put_quote(adapter.name, leg, quote, ttl)
            self._record_delta(leg, quote, adapter.name)
        self._check_floor(leg, quote, ctx)
        return quote

    def _prices_a_real_ticket(self, leg: Leg) -> bool:
        """Freerider and walking have no ticket, so their floor carries no fare information.

        A Freerider "floor" is produced by the same cost model that produces its quote, so
        comparing the two would only measure the model against itself.
        """
        return leg.mode not in (TransportMode.FREERIDER, TransportMode.WALK)

    def _check_floor(self, leg: Leg, quote: Quote, ctx: _Context) -> None:
        """A floor above the true price means McRAPTOR could have pruned the cheapest trip."""
        if not quote.is_priced or quote.amount_ore is None or not self._prices_a_real_ticket(leg):
            return
        if quote.source == PASS_SOURCE:
            # A travel card zeroes the leg for the holder; that is below the floor by design,
            # not a routing bug, so it is not a floor violation.
            return
        floor = self.floors.floor_ore(leg.mode, leg.operator, leg_distance_km(leg))
        if floor > quote.amount_ore:
            ctx.violations.append(
                f"{leg.operator or leg.mode.value}: floor {floor} > actual {quote.amount_ore}"
            )

    def _record_delta(self, leg: Leg, quote: Quote, source: str) -> None:
        """Log (floor, actual) so the routing bound can be calibrated and audited."""
        if self.db is None or not quote.is_priced or quote.amount_ore is None:
            return
        if not self._prices_a_real_ticket(leg) or source == PASS_SOURCE:
            return
        distance = leg_distance_km(leg)
        floor = self.floors.floor_ore(leg.mode, leg.operator, distance)
        self.db.record_reprice_delta(
            source=source,
            mode=leg.mode.value,
            operator=leg.operator,
            distance_km=distance,
            floor_ore=floor,
            actual_ore=quote.amount_ore,
        )

    # --- one itinerary ----------------------------------------------------

    async def _price_itinerary(self, itin: Itinerary, ctx: _Context) -> Itinerary:
        vehicle_legs = [leg for leg in itin.legs if leg.mode is not TransportMode.WALK]
        quotes = await asyncio.gather(*(self._quote_leg(leg, ctx) for leg in vehicle_legs))

        priced: list[Leg] = []
        quote_iter = iter(quotes)
        for leg in itin.legs:
            if leg.mode is TransportMode.WALK:
                priced.append(leg)
                continue
            priced.append(leg.model_copy(update={"quote": next(quote_iter)}))

        warnings = list(itin.warnings)
        warnings.extend(self._warnings_for(priced))
        return Itinerary(
            legs=priced, warnings=warnings, floor_price_ore=itin.floor_price_ore
        )

    def _warnings_for(self, legs: list[Leg]) -> list[str]:
        warnings: list[str] = []
        freerider = next(
            (a for a in self.adapters if isinstance(a, FreeriderAdapter)), None
        )
        for leg in legs:
            if leg.mode is TransportMode.FREERIDER and freerider is not None:
                warnings.extend(freerider.warnings_for(leg))
            elif leg.mode is TransportMode.LOCAL_TRANSIT:
                warnings.append(
                    f"{leg.from_stop.name} to {leg.to_stop.name} is local transit; "
                    "buy a local ticket separately, its price is not included."
                )
            quote = leg.quote
            if quote is None:
                continue
            if quote.confidence is PriceConfidence.STALE:
                warnings.append(
                    f"Price for the {leg.operator or leg.mode.value} leg is stale "
                    "and may have changed."
                )
            elif quote.confidence is PriceConfidence.UNAVAILABLE and leg.is_transit:
                warnings.append(
                    f"Could not price the {leg.operator or leg.mode.value} leg "
                    f"{leg.from_stop.name} to {leg.to_stop.name}."
                )
        return warnings

    # --- public entry point ----------------------------------------------

    async def price(
        self,
        candidates: list[Itinerary],
        constraints: SearchConstraints | None = None,
        *,
        max_results: int = 5,
        require_priced: bool = True,
    ) -> PricingResult:
        """Price the most promising candidates, then rank by true total price.

        With `require_priced`, itineraries containing a leg we could not price are dropped
        rather than shown ranked below the priced ones: an option whose real cost is unknown
        is not a "cheapest" answer, and a list full of "price unavailable" rows is noise. The
        drop is not absolute - if it would leave nothing, the unpriced options come back with
        a notice, because an empty result on a route we *can* route is worse than an honest
        "we can't price this, here's the booking link".
        """
        constraints = constraints or SearchConstraints()
        feasible = [itin for itin in candidates if constraints.permits(itin)]
        if not feasible:
            return PricingResult(itineraries=[], warnings=["No itinerary met the constraints."])

        pre_warnings: list[str] = []
        if require_priced:
            # Drop itineraries with a leg no adapter can price *before* spending any calls.
            # An itinerary that rides an operator with no price source (Skanetrafiken,
            # Oresundstag) can never be fully priced, so pricing its other legs is wasted
            # work - and on a route dominated by such operators, that waste is what makes the
            # whole search time out. Structural filter first, runtime filter after.
            priceable = [itin for itin in feasible if self._itinerary_priceable(itin)]
            if priceable:
                if len(priceable) < len(feasible):
                    pre_warnings.append(
                        f"{len(feasible) - len(priceable)} route(s) skipped: they use an "
                        "operator with no available price source."
                    )
                feasible = priceable

        # The range query returns one itinerary per departure, so the same journey shape
        # appears many times. Sample each shape across the day before spending calls: the
        # cheapest departure is often the last one, never the first.
        feasible = spread_by_departure(feasible, self.budget.max_departures_per_pattern)
        to_price, deferred = _select_to_price(feasible, self.budget.max_candidates_to_price)

        ctx = _Context(call_budget=CallBudget.from_settings(self.budget))
        for adapter in self.adapters:
            setter = getattr(adapter, "set_budget", None)
            if setter is not None:
                setter(ctx.call_budget)
        warnings: list[str] = list(pre_warnings)

        try:
            async with asyncio.timeout(self.budget.phase2_timeout_seconds):
                priced = await asyncio.gather(
                    *(self._price_itinerary(itin, ctx) for itin in to_price)
                )
        except TimeoutError:
            warnings.append(
                "Pricing timed out; some itineraries are shown with incomplete prices."
            )
            priced = to_price

        if deferred:
            warnings.append(
                f"{len(deferred)} slower or pricier itineraries were not priced "
                f"(cap: {self.budget.max_candidates_to_price})."
            )

        if ctx.violations:
            # The router's bound exceeded a real fare, so a cheaper journey may never have
            # reached this function. Say so rather than present the ranking as authoritative.
            warnings.append(
                f"{len(ctx.violations)} price-floor violation(s) detected "
                f"({ctx.violations[0]}); a cheaper itinerary may have been pruned "
                "before pricing."
            )

        for adapter in self.adapters:
            setter = getattr(adapter, "set_budget", None)
            if setter is not None:
                setter(None)

        ranked = sorted(collapse_equivalent(priced), key=_price_sort_key)

        if require_priced:
            fully = [itin for itin in ranked if itin.fully_priced]
            if fully:
                dropped = len(ranked) - len(fully)
                ranked = fully
                if dropped:
                    warnings.append(
                        f"{dropped} option(s) with at least one unpriced leg were hidden."
                    )
            else:
                warnings.append(
                    "No fully-priced route was found. Showing options with unpriced legs; "
                    "check those fares on the operator's own site."
                )

        # Every selected travel mode should be represented, not just whichever is cheapest.
        # Take the top-N by price, then pull in the cheapest itinerary using each allowed mode
        # that the top-N missed, so ticking train+bus+car shows a train, a bus and a car option
        # rather than a list of whichever mode happened to win on price.
        shown = _ensure_mode_coverage(ranked, _cover_modes(constraints), max_results)

        status = {a.name: (await a.health()).state.value for a in self.adapters}

        return PricingResult(
            itineraries=shown,
            warnings=warnings + _collect_warnings(shown),
            source_status=status,
            calls_made=ctx.call_budget.spent(),
            floor_violations=len(ctx.violations),
        )


#: The user-facing travel modes a "Travel by" tick can select. Walk and local transit are
#: connective, not a choice, so they never need their own representative.
_TRAVEL_MODES = (
    TransportMode.TRAIN,
    TransportMode.BUS,
    TransportMode.FERRY,
    TransportMode.FREERIDER,
    TransportMode.FLIGHT,
)


def _cover_modes(constraints: SearchConstraints) -> list[TransportMode]:
    """The selected travel modes to guarantee a representative for, cheapest-mode-first ish."""
    return [m for m in _TRAVEL_MODES if m in constraints.allowed_modes]


def _itinerary_modes(itin: Itinerary) -> set[TransportMode]:
    return {leg.mode for leg in itin.legs if leg.mode is not TransportMode.WALK}


def _ensure_mode_coverage(
    ranked: list[Itinerary], cover_modes: list[TransportMode], max_results: int
) -> list[Itinerary]:
    """Top-N by price, plus the cheapest priced itinerary for each mode the top-N missed.

    `ranked` is already price-sorted (unpriced sink to the bottom). The top-N stands; then for
    each selected mode not yet represented, the first (cheapest) priced itinerary using it is
    appended. A mode with no priced option adds nothing. The result is re-sorted by price, so
    a guaranteed train representative still lands in its correct price position.
    """
    shown = list(ranked[:max_results])
    covered: set[TransportMode] = set()
    for itin in shown:
        covered |= _itinerary_modes(itin)

    chosen_ids = {id(i) for i in shown}
    for mode in cover_modes:
        if mode in covered:
            continue
        for itin in ranked:
            if itin.total_price_ore is None:
                break  # ranked is price-sorted; once we hit unpriced, none are priced
            if id(itin) in chosen_ids:
                continue
            if mode in _itinerary_modes(itin):
                shown.append(itin)
                chosen_ids.add(id(itin))
                covered |= _itinerary_modes(itin)
                break

    return sorted(shown, key=_price_sort_key)


def _floor_sort_key(itin: Itinerary) -> tuple:
    """Order by the router's price lower bound, then by arrival.

    The bound comes from McRAPTOR, not from any quote: at this point nothing is priced.
    A candidate with the lowest bound is the best guess at the eventual cheapest, so it is
    the one worth spending a scarce upstream call on.
    """
    return (itin.floor_price_ore or 0, itin.arrival, itin.transfers)


def _extremes_first(group: list[Itinerary]) -> list[Itinerary]:
    """Reorder one journey shape's departures as earliest, latest, then inward.

    Within a shape every departure carries the same price floor, so the router gives no
    hint which is cheapest. The two most informative probes are the ends of the day: the
    07:30 coach costs 800 SEK and the 22:55 costs 420. Ordering by departure and taking the
    head - which is what any floor-then-arrival sort does - probes the expensive end twice
    and never reaches the cheap one when the budget is tight.
    """
    remaining = sorted(group, key=lambda i: i.departure)
    ordered: list[Itinerary] = []
    while remaining:
        ordered.append(remaining.pop(0))
        if remaining:
            ordered.append(remaining.pop())
    return ordered


def _select_to_price(
    candidates: list[Itinerary], limit: int
) -> tuple[list[Itinerary], list[Itinerary]]:
    """Choose which candidates to price: round-robin across shapes, extremes first within one.

    Two biases have to be defeated at once. Across shapes, a cheap-floor shape should be
    probed before an expensive one. Within a shape, the cheapest departure is as likely to be
    the last of the day as the first, so the ends are probed before the middle.
    """
    if len(candidates) <= limit:
        return sorted(candidates, key=_floor_sort_key), []

    groups: dict[tuple, list[Itinerary]] = {}
    for itin in candidates:
        groups.setdefault(pattern_key(itin), []).append(itin)

    ordered = sorted(
        (_extremes_first(group) for group in groups.values()),
        key=lambda g: _floor_sort_key(min(g, key=_floor_sort_key)),
    )

    chosen: list[Itinerary] = []
    depth = 0
    while len(chosen) < limit and any(len(g) > depth for g in ordered):
        for group in ordered:
            if depth < len(group):
                chosen.append(group[depth])
                if len(chosen) == limit:
                    break
        depth += 1

    picked = {id(itin) for itin in chosen}
    deferred = [itin for itin in candidates if id(itin) not in picked]
    return chosen, deferred


def _price_sort_key(itin: Itinerary) -> tuple:
    """Cheapest first. Unpriced itineraries sink to the bottom.

    An itinerary missing a leg price must never outrank a complete one just because the
    hole summed to nothing.
    """
    total = itin.total_price_ore
    return (
        0 if total is not None else 1,
        total if total is not None else 0,
        _CONFIDENCE_RANK[itin.price_confidence],
        itin.duration_seconds,
        itin.transfers,
    )


def _collect_warnings(itineraries: list[Itinerary]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for itin in itineraries:
        for warning in itin.warnings:
            if warning not in seen:
                seen.add(warning)
                out.append(warning)
    return out


def utcnow() -> datetime:
    return datetime.now(UTC)
