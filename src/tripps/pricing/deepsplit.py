"""Exhaustive split-ticket scan of a single SJ train.

`split.py` asks a cheap question during a normal search: "is one break, at one of twelve
curated hubs, cheaper than the through fare?" This module asks the expensive one, on demand:
**what is the cheapest set of tickets that covers this exact train, breaking anywhere it
calls, as many times as it takes?**

Why that is a different question. SJ prices every origin/destination pair independently and
allocates cheap seats per segment, so the through fare is not the sum of anything - it is its
own bucket. Measured on 2026-08-10, Stockholm C -> Malmö C: the 07:19 through fare is 1735
SEK while Stockholm->Alvesta + Alvesta->Malmö on that same train is 1405 + 195 = 1600. The
saving lives on the *expensive* departures, which a cheapest-first search never shows, so
during a normal search the finding is usually invisible. Here the traveller has already
chosen the train and wants the best way to pay for it.

The optimisation is exact rather than greedy. With a price for every pair of calling points,
the cheapest ticket chain is a shortest path through a DAG whose nodes are the stops and
whose edge (i, j) is the fare from stop i to stop j - so `best_chain` returns the true
optimum over *every* number of breaks, not the best single break. What bounds the work is
the pricing, not the search: one pair costs an SJ day-search plus an offer lookup, so an
n-stop train is O(n^2) pairs. `max_points` caps how many calling points are scanned, and
whatever it drops is reported rather than quietly folded away - a scan that silently skipped
half the train would present a merely-good chain as the cheapest one.

Two contracts are not one. A chain of tickets carries no rebooking or delay protection across
a break, so a chain must beat the through fare by `MIN_SAVING_ORE` before it is called
cheaper; on a tie, the single ticket wins. When SJ sells no through fare at all, any complete
chain is reported as what it is: the only way to buy this train.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from ..models import ADULT, Leg, Passenger, Stop, TransportMode
from ..routing.timetable import Timetable
from ..timeutil import from_service_seconds
from .base import CallBudget
from .sj import SJ_OPERATORS

log = logging.getLogger(__name__)

#: A chain must beat the through fare by at least this to be called cheaper. Two tickets mean
#: two contracts; a handful of kronor does not pay for losing delay protection at the break.
MIN_SAVING_ORE = 2_000

#: Default ceiling on calling points scanned. Pairs grow quadratically and each costs ~3
#: upstream calls, so 12 points is ~66 pairs is ~200 calls - minutes, not hours, and already
#: far beyond what a normal search may spend.
DEFAULT_MAX_POINTS = 12

#: Call allowance for one scan. Deliberately large: this is user-initiated, runs on its own
#: budget, and is the whole point of the command.
DEFAULT_SCAN_CALLS = 400


@dataclass(frozen=True, slots=True)
class CallingPoint:
    """One stop this train calls at, with whether tickets may start or end there."""

    stop: Stop
    departure: datetime
    arrival: datetime
    can_board: bool
    can_alight: bool


@dataclass(frozen=True, slots=True)
class TrainRun:
    """One physical train, from the traveller's boarding stop to their alighting stop."""

    trip_id: str
    operator: str | None
    headsign: str | None
    points: tuple[CallingPoint, ...]

    @property
    def departure(self) -> datetime:
        return self.points[0].departure

    @property
    def arrival(self) -> datetime:
        return self.points[-1].arrival

    @property
    def origin(self) -> Stop:
        return self.points[0].stop

    @property
    def destination(self) -> Stop:
        return self.points[-1].stop

    def label(self) -> str:
        return (
            f"{self.departure:%H:%M} {self.origin.name} -> "
            f"{self.arrival:%H:%M} {self.destination.name}"
            f" ({self.operator or 'unknown'} {self.trip_id})"
        )


@dataclass(frozen=True, slots=True)
class Ticket:
    from_point: CallingPoint
    to_point: CallingPoint
    price_ore: int


@dataclass(slots=True)
class ScanResult:
    run: TrainRun
    through_ore: int | None
    tickets: list[Ticket]
    scanned_points: int
    total_points: int
    pairs_priced: int
    pairs_unsellable: int
    calls_made: int
    notes: list[str] = field(default_factory=list)

    @property
    def chain_ore(self) -> int | None:
        if not self.tickets:
            return None
        return sum(t.price_ore for t in self.tickets)

    @property
    def saving_ore(self) -> int | None:
        """How much the chain beats the through fare by, or None if there is nothing to beat."""
        chain = self.chain_ore
        if chain is None or self.through_ore is None:
            return None
        return self.through_ore - chain

    @property
    def chain_wins(self) -> bool:
        """Is the chain the answer? A single contract wins ties, and near-ties.

        Two readings of "cheaper" collapse here on purpose: a chain that undercuts a real
        through fare by a worthwhile margin, and a chain that exists where no through fare
        does at all. Both mean "buy these tickets"; neither means "buy them to save 4 kronor".
        """
        if self.chain_ore is None:
            return False
        if self.through_ore is None:
            return True
        return self.through_ore - self.chain_ore >= MIN_SAVING_ORE


# --- finding the train -----------------------------------------------------


def direct_runs(
    timetable: Timetable,
    origin_ids: set[str],
    destination_ids: set[str],
    service_date: date,
    *,
    operators: frozenset[str] = SJ_OPERATORS,
) -> list[TrainRun]:
    """Every train running directly from one of `origin_ids` to one of `destination_ids`.

    Straight off the timetable rather than through the planner: the traveller has named a
    train, so routing, pricing and ranking a whole day of alternatives would be work whose
    result is thrown away. Boarding restrictions travel with each calling point, because a
    set-down-only stop cannot be a place a ticket starts.
    """
    origin_idx = {timetable.stop_index[s] for s in origin_ids if s in timetable.stop_index}
    dest_idx = {timetable.stop_index[s] for s in destination_ids if s in timetable.stop_index}
    if not origin_idx or not dest_idx:
        return []

    runs: list[TrainRun] = []
    for route in timetable.routes:
        if route.info.mode is not TransportMode.TRAIN:
            continue
        if route.info.operator not in operators:
            continue
        board_pos = next((i for i, s in enumerate(route.stops) if s in origin_idx), None)
        if board_pos is None:
            continue
        alight_pos = next(
            (
                i
                for i in range(len(route.stops) - 1, board_pos, -1)
                if route.stops[i] in dest_idx
            ),
            None,
        )
        if alight_pos is None:
            continue

        for trip in route.trips:
            points = []
            for pos in range(board_pos, alight_pos + 1):
                points.append(
                    CallingPoint(
                        stop=timetable.stops[route.stops[pos]],
                        departure=from_service_seconds(trip.departures[pos], service_date),
                        arrival=from_service_seconds(trip.arrivals[pos], service_date),
                        can_board=not (trip.no_board or ())[pos] if trip.no_board else True,
                        can_alight=not (trip.no_alight or ())[pos] if trip.no_alight else True,
                    )
                )
            runs.append(
                TrainRun(
                    trip_id=trip.id,
                    operator=route.info.operator,
                    headsign=trip.headsign,
                    points=tuple(points),
                )
            )
    runs.sort(key=lambda r: r.departure)
    return runs


@dataclass(frozen=True, slots=True)
class PointSelection:
    """Which calling points a scan will cover, and what it left out - and why.

    The two reasons are kept apart because only one of them is the user's to change. A stop
    the operator lets nobody board at can never bound a ticket however high `--max-points`
    goes; a stop dropped to stay under the cap is a coverage choice the traveller can undo.
    Reporting both as one number would invite raising a limit that changes nothing.
    """

    indices: list[int]
    #: Interior stops that cannot bound a ticket: set-down-only or pick-up-only.
    unusable: int
    #: Interior stops dropped purely to stay within `max_points`.
    capped: int


def choose_points(run: TrainRun, max_points: int) -> PointSelection:
    """The calling points to scan: both endpoints, plus interior ones, capped.

    The endpoints are mandatory - they are what the traveller asked for. Interior points are
    thinned evenly along the train when there are too many, because breaks that pay off tend
    to sit where fare zones change rather than clustered at one end, and an even spread is the
    least-assuming way to sample without knowing where those are.
    """
    interior = range(1, len(run.points) - 1)
    usable = [i for i in interior if run.points[i].can_alight and run.points[i].can_board]
    unusable = len(interior) - len(usable)

    interior_budget = max(max_points - 2, 0)
    kept = usable
    if len(usable) > interior_budget:
        if interior_budget == 0:
            kept = []
        else:
            step = len(usable) / interior_budget
            kept = [usable[int(k * step)] for k in range(interior_budget)]
    return PointSelection(
        indices=[0, *kept, len(run.points) - 1],
        unusable=unusable,
        capped=len(usable) - len(kept),
    )


# --- pricing ---------------------------------------------------------------


def _sub_leg(run: TrainRun, i: int, j: int) -> Leg:
    """The synthetic leg for riding this same train from calling point i to j."""
    points = run.points[i : j + 1]
    return Leg(
        from_stop=points[0].stop,
        to_stop=points[-1].stop,
        mode=TransportMode.TRAIN,
        operator=run.operator,
        departure=points[0].departure,
        arrival=points[-1].arrival,
        service_ref=run.trip_id,
        headsign=run.headsign,
        via_stop_ids=tuple(p.stop.id for p in points),
        via_departures=tuple(p.departure for p in points),
        via_arrivals=tuple(p.arrival for p in points),
    )


async def scan_train(
    adapter,
    run: TrainRun,
    *,
    passenger: Passenger = ADULT,
    max_points: int = DEFAULT_MAX_POINTS,
    max_calls: int = DEFAULT_SCAN_CALLS,
    progress=None,
) -> ScanResult:
    """Price every pair of scanned calling points, then solve for the cheapest ticket chain.

    Runs on its own `CallBudget`: this is an explicit, user-initiated scan, and the per-search
    allowance exists to stop a *background* fan-out, not this. Pairs that come back unpriced
    (SJ sells no ticket for that pair on this train) are simply absent edges - the chain routes
    around them, which is what makes the scan useful when a through fare is missing entirely.
    """
    selection = choose_points(run, max_points)
    chosen = selection.indices
    budget = CallBudget(limit=max_calls)
    previous_budget = getattr(adapter, "_budget", None)
    setter = getattr(adapter, "set_budget", None)
    if setter is not None:
        setter(budget)

    prices: dict[tuple[int, int], int] = {}
    unsellable = 0
    notes: list[str] = []
    try:
        for a in range(len(chosen)):
            for b in range(a + 1, len(chosen)):
                i, j = chosen[a], chosen[b]
                if not run.points[i].can_board or not run.points[j].can_alight:
                    continue
                try:
                    quote = await adapter.quote_leg(_sub_leg(run, i, j), passenger)
                except Exception as exc:  # noqa: BLE001 - one dead pair must not end the scan
                    log.warning("split scan: %s->%s failed: %s", i, j, exc)
                    notes.append(
                        f"{run.points[i].stop.name} -> {run.points[j].stop.name}: "
                        f"lookup failed ({type(exc).__name__})"
                    )
                    continue
                if quote.amount_ore is None or not quote.is_priced:
                    unsellable += 1
                    continue
                prices[(a, b)] = quote.amount_ore
                if progress is not None:
                    progress(run.points[i].stop.name, run.points[j].stop.name, quote.amount_ore)
    finally:
        if setter is not None:
            setter(previous_budget)

    total_points = len(run.points)
    if selection.capped:
        notes.append(
            f"{selection.capped} of this train's {total_points} calling points were not "
            f"scanned, to stay within --max-points {max_points}; a break at one of them was "
            "not tested. Raise it to scan further."
        )
    if selection.unusable:
        notes.append(
            f"{selection.unusable} calling point(s) cannot be a break: the operator lets "
            "nobody board or nobody alight there, so no ticket can start or end at them."
        )

    through = prices.get((0, len(chosen) - 1))
    tickets = [
        Ticket(run.points[chosen[a]], run.points[chosen[b]], price)
        for a, b, price in best_chain(prices, len(chosen))
    ]
    return ScanResult(
        run=run,
        through_ore=through,
        tickets=tickets,
        scanned_points=len(chosen),
        total_points=total_points,
        pairs_priced=len(prices),
        pairs_unsellable=unsellable,
        calls_made=budget.spent(),
        notes=notes,
    )


# --- the optimisation ------------------------------------------------------


def best_chain(prices: dict[tuple[int, int], int], n: int) -> list[tuple[int, int, int]]:
    """The cheapest chain of tickets from node 0 to node n-1, as (from, to, price) triples.

    A shortest path over a DAG, so it is the exact optimum across every number of breaks
    rather than the best single split: `cost[j]` is the cheapest way to have reached j, and
    every edge runs strictly forward, so one pass in increasing j settles each node. Returns
    an empty list when no chain of sellable tickets spans the train.
    """
    if n < 2:
        return []
    infinity = float("inf")
    cost: list[float] = [infinity] * n
    came_from: list[tuple[int, int] | None] = [None] * n
    cost[0] = 0
    for j in range(1, n):
        for i in range(j):
            price = prices.get((i, j))
            if price is None or cost[i] == infinity:
                continue
            candidate = cost[i] + price
            if candidate < cost[j]:
                cost[j] = candidate
                came_from[j] = (i, price)
    if cost[n - 1] == infinity:
        return []

    chain: list[tuple[int, int, int]] = []
    node = n - 1
    while node != 0:
        step = came_from[node]
        assert step is not None  # cost[node] is finite, so it was reached from somewhere
        previous, price = step
        chain.append((previous, node, price))
        node = previous
    chain.reverse()
    return chain
