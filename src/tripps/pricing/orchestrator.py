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
    SEK,
    Itinerary,
    Leg,
    PriceConfidence,
    Quote,
    SearchConstraints,
    TransportMode,
)
from ..passes import (
    TICKITAL_FARE_CLASS,
    TICKITAL_SOURCE,
    PassAdapter,
    TickitalAdapter,
    tickital_note,
)
from ..routing.floors import PriceFloorModel
from ..routing.journey import (
    collapse_equivalent,
    leg_distance_km,
    pattern_key,
    spread_by_departure,
)
from ..surcharges import apply_arlanda_fee
from .base import BudgetExceeded, CallBudget
from .freerider import FreeriderAdapter
from .sj import SOURCE as SJ_SOURCE
from .split import sub_legs as _split_sub_legs

log = logging.getLogger(__name__)

#: Quote source used by the travel-card pass adapter; recognised here so a card-covered free
#: leg is not mistaken for a price-floor violation.
PASS_SOURCE = "travelcard"

#: Sources whose amount is below the routing floor by design - an owned card zeroes a leg, a
#: tickital rental fallback carries a whole period price on one leg - so neither is a floor
#: violation nor a fare sample for calibration.
_FLOOR_EXEMPT_SOURCES = frozenset({PASS_SOURCE, TICKITAL_SOURCE})

#: Confidences firm enough to decide "the rental beats the singles". A stale or estimated
#: single is too soft to justify pushing a traveller into a terms-risky rental.
_FIRM_CONFIDENCES = frozenset({PriceConfidence.EXACT, PriceConfidence.CACHED})

#: Don't trumpet a split saving below this - the two-contract hassle is not worth a few kronor,
#: and a saving that small is within the noise of a re-priced yield-managed fare.
_SPLIT_MIN_SAVING_ORE = 500


def _splittable(leg: Leg) -> bool:
    """An SJ leg firmly enough priced, with a known stop path, to test a split against."""
    q = leg.quote
    return (
        q is not None
        and q.source == SJ_SOURCE
        and q.amount_ore is not None
        and q.confidence in _FIRM_CONFIDENCES
        and q.split_hint is None
        and bool(leg.via_stop_ids)
    )

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
    #: Operators whose routing floor the search zeroed (held cards + active rentals). A leg on
    #: one of these was routed with a 0 floor, so a paid quote below the calibrated floor
    #: pruned nothing and must not be reported as a floor violation.
    zeroed_operators: frozenset[str] = frozenset()


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
        if quote.source in _FLOOR_EXEMPT_SOURCES or leg.operator in ctx.zeroed_operators:
            # A travel card zeroes the leg for the holder, and a tickital rental fallback
            # carries a whole period price on one leg; both are below (or unlike) the per-leg
            # floor by design, not a routing bug. Likewise an operator whose routing floor the
            # search zeroed (a card/rental honours it) was routed with a 0 lower bound, so a
            # paid quote below the calibrated floor pruned nothing - not a violation.
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
        if not self._prices_a_real_ticket(leg) or source in _FLOOR_EXEMPT_SOURCES:
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

    # --- split ticketing (advisory) ---------------------------------------

    async def _price_sub_leg(self, leg: Leg) -> Quote:
        """Price a synthetic split sub-leg. Like `_quote_leg`, but never records a floor
        violation or a calibration sample: a sub-leg was not routed, so its fare says nothing
        about the routing bound, and a spurious violation would wrongly alarm the traveller."""
        adapter = self._adapter_for(leg)
        if adapter is None:
            return Quote.unavailable(source="none")
        if self.db is not None:
            cached = self.db.get_quote(adapter.name, leg)
            if cached is not None and cached.confidence in (
                PriceConfidence.CACHED,
                PriceConfidence.ESTIMATED,
            ):
                return cached
        try:
            quote = await adapter.quote_leg(leg)
        except BudgetExceeded as exc:
            return Quote.unavailable(source=adapter.name, note=str(exc))
        except Exception as exc:  # noqa: BLE001 - a split probe must never empty the results
            log.exception("split sub-leg pricing failed on %s", leg.service_ref)
            return Quote.unavailable(source=adapter.name, note=f"pricing failed: {exc}")
        if self.db is not None and quote.is_priced:
            self.db.put_quote(adapter.name, leg, quote, self.ttl.for_mode(leg.mode))
        return quote

    async def _best_split_hint(self, leg: Leg) -> str | None:
        """The cheapest firm split of this SJ leg, as advisory text, or None if none undercuts
        the through fare by a worthwhile margin. Both halves must be firmly priced on the same
        train, so the saving is one a traveller can actually realise."""
        through = leg.quote.amount_ore
        best_total: int | None = None
        best_name = ""
        for name, first, second in _split_sub_legs(leg):
            q1 = await self._price_sub_leg(first)
            if q1.amount_ore is None or q1.confidence not in _FIRM_CONFIDENCES:
                continue
            q2 = await self._price_sub_leg(second)
            if q2.amount_ore is None or q2.confidence not in _FIRM_CONFIDENCES:
                continue
            total = q1.amount_ore + q2.amount_ore
            if best_total is None or total < best_total:
                best_total, best_name = total, name
        if best_total is None or through - best_total < _SPLIT_MIN_SAVING_ORE:
            return None
        saving = through - best_total
        return (
            f"Split at {best_name}: two separate SJ tickets total {best_total / SEK:.0f} kr, "
            f"{saving / SEK:.0f} kr less than the {through / SEK:.0f} kr through fare "
            f"(two tickets, no rebooking or delay protection across {best_name})."
        )

    async def _apply_split_tickets(self, shown: list[Itinerary]) -> None:
        """Annotate SJ legs in the shown itineraries where a hub split beats the through fare.

        Advisory only: neither `amount_ore` nor the itinerary total moves - the price tripps
        stands behind stays the bookable through fare. Each distinct train is probed once (many
        itineraries reuse it) and the hint is copied onto every leg that rides it.

        Runs on its OWN fresh budget, not the search's: main pricing routinely spends the SJ
        call allowance, and sharing it would leave the split probe unable to price either half
        and silently find nothing. A separate allowance keeps this advisory bounded on its own
        terms while never starving the authoritative pricing that precedes it.
        """
        if not self.budget.enable_split_tickets:
            return
        split_budget = CallBudget.from_settings(self.budget)
        for adapter in self.adapters:
            setter = getattr(adapter, "set_budget", None)
            if setter is not None:
                setter(split_budget)
        try:
            seen: dict[tuple[str, str, str], str | None] = {}
            for itin in shown:
                for i, leg in enumerate(itin.legs):
                    if not _splittable(leg):
                        continue
                    key = (leg.from_stop.id, leg.to_stop.id, leg.departure.isoformat())
                    if key not in seen:
                        seen[key] = await self._best_split_hint(leg)
                    hint = seen[key]
                    if hint is not None:
                        itin.legs[i] = leg.model_copy(
                            update={"quote": leg.quote.model_copy(update={"split_hint": hint})}
                        )
        finally:
            for adapter in self.adapters:
                setter = getattr(adapter, "set_budget", None)
                if setter is not None:
                    setter(None)

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

        # Reconcile the whole itinerary now that every leg has a quote: fold a tickital
        # rental into one charge, then add the Arlanda C passage fee (which a period pass does
        # not cover) on top. Both run after per-leg pricing and before warnings.
        priced = self._apply_rental_coupon(priced)
        priced, arlanda_warning = apply_arlanda_fee(priced)

        warnings = list(itin.warnings)
        if arlanda_warning is not None:
            warnings.append(arlanda_warning)
        warnings.extend(self._warnings_for(priced))
        return Itinerary(
            legs=priced, warnings=warnings, floor_price_ore=itin.floor_price_ore
        )

    def _apply_rental_coupon(self, priced: list[Leg]) -> list[Leg]:
        """Charge a tickital rental once per itinerary, and only when it beats the singles.

        Covered legs arrive here already priced - by a real paid single, or (when no paid
        source could) by the `TickitalAdapter` fallback whose amount is the rental's period
        price. For each rental, group the legs it covers, then decide:

        * if any covered leg has no purchasable single - a tickital fallback quote, or a paid
          operator that returned UNAVAILABLE - the rental is the ONLY complete pricing, so it
          is applied regardless of price and of the other legs' confidence;
        * otherwise apply the rental only when its period price is *strictly* below the sum of
          the covered legs' singles AND those singles are firm (EXACT/CACHED); a stale or
          estimated single is too soft to push a traveller into a terms-risky rental.

        When applied, the period price lands once on the first covered leg and the rest are
        zeroed, so the naive per-leg total sums to exactly one rental price. Legs an owned card
        already zeroed (source == PASS_SOURCE) are skipped, so a rental never re-charges a leg
        a card covers, and each leg lands in at most one rental group (first match wins).
        """
        tickital = next((a for a in self.adapters if isinstance(a, TickitalAdapter)), None)
        if tickital is None or not tickital.has_rentals():
            return priced

        groups: dict[object, list[int]] = {}
        cards: dict[object, tuple] = {}
        for i, leg in enumerate(priced):
            if leg.mode is TransportMode.WALK:
                continue
            quote = leg.quote
            if quote is not None and quote.source == PASS_SOURCE:
                continue  # already free via an owned card; never fold into a rental group
            match = tickital.rental_for(leg)
            if match is None:
                continue
            card, rental = match
            key = rental.id if rental.id is not None else id(rental)
            groups.setdefault(key, []).append(i)
            cards[key] = (card, rental)
        if not groups:
            return priced

        result = list(priced)
        for key, idxs in groups.items():
            card, rental = cards[key]
            single_idx = [
                j
                for j in idxs
                if (q := result[j].quote) is not None
                and q.is_priced
                and q.source != TICKITAL_SOURCE
            ]
            has_fallback = len(single_idx) < len(idxs)  # a covered leg with no purchasable single
            firm = all(result[j].quote.confidence in _FIRM_CONFIDENCES for j in single_idx)
            singles_sum = sum(result[j].quote.amount_ore for j in single_idx)

            if has_fallback:
                apply = True
            else:
                apply = firm and rental.price_ore < singles_sum
            if not apply:
                continue

            note = tickital_note(card, rental)
            for pos, j in enumerate(idxs):
                result[j] = result[j].model_copy(
                    update={
                        "quote": Quote(
                            source=TICKITAL_SOURCE,
                            amount_ore=rental.price_ore if pos == 0 else 0,
                            confidence=PriceConfidence.EXACT,
                            fare_class=TICKITAL_FARE_CLASS,
                            note=note if pos == 0 else None,
                            coupon_rental_id=rental.id,
                        )
                    }
                )
        return result

    def _warnings_for(self, legs: list[Leg]) -> list[str]:
        warnings: list[str] = []
        freerider = next(
            (a for a in self.adapters if isinstance(a, FreeriderAdapter)), None
        )
        passcard = next((a for a in self.adapters if isinstance(a, PassAdapter)), None)
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
            # A held card that partially covers this leg (crosses its county border): we charge
            # the full fare because zone-combination pricing is not modelled, but say the card
            # may reduce it. Only when the leg is actually charged (not already free via a pass).
            if (
                passcard is not None
                and quote.is_priced
                and quote.source not in (PASS_SOURCE, TICKITAL_SOURCE)
            ):
                partial = passcard.partial_card(leg)
                if partial is not None:
                    warnings.append(
                        f"Your {partial.name} card may reduce the {leg.operator or leg.mode.value} "
                        f"fare {leg.from_stop.name} to {leg.to_stop.name}: it crosses your card's "
                        "county border, and cross-border zone-combination pricing is not modelled, "
                        "so the leg is shown at full fare."
                    )
            if quote.fare_class == TICKITAL_FARE_CLASS and quote.note:
                # After the coupon, only the one charged leg carries a note (its zeroed
                # siblings get note=None), so this fires exactly once per rental. The note
                # already states the one-time cost and the terms risk.
                warnings.append(quote.note)
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
        zeroed_operators: frozenset[str] | None = None,
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

        ctx = _Context(
            call_budget=CallBudget.from_settings(self.budget),
            zeroed_operators=zeroed_operators or frozenset(),
        )
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

        # On the handful of itineraries actually shown, flag any SJ leg a hub split undercuts.
        # Advisory-only and budget-bounded, so it neither reorders the results nor risks the
        # cheapest being missed - it just tells the traveller how to shave the fare further.
        await self._apply_split_tickets(shown)

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
