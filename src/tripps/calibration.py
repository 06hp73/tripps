"""Calibrate the router's price floors from observed fares.

McRAPTOR optimizes a price *lower bound*, not the real fare, because the real fare is only
knowable per concrete itinerary. `routing/floors.py` ships hand-set defaults ("train >= 19
SEK + 5 öre/km") chosen to be safely below any possible fare, which is correct but far too
loose: a slack bound bloats the Pareto frontier, so the router carries more candidate
itineraries than it needs and phase 2 spends more upstream calls pricing them.

Every time phase 2 prices a leg it already logs the (floor, actual) pair to `reprice_delta`.
This module closes that loop: it reads those observations and fits a tighter per-operator
floor, then stores it so the next search's router prunes better.

All distances here are RIDDEN-PATH kilometres (`Leg.path_km`: the sum of the route's
per-segment great-circle hops) - the same polyline McRAPTOR integrates `per_km` over. Fitting
against the shorter endpoint straight-line and applying over the path would break the floor
contract on winding routes; `db._migrate` wipes any samples recorded under that older metric.

The one hard constraint is the floor's correctness contract, unchanged from floors.py:

    floor(leg) <= true_price(leg),  always.

A floor above the truth can prune the genuinely cheapest itinerary before it is priced. So
the fit is provably safe on the observed data and then discounted by a margin for headroom:

    per_km = min over observations of actual / distance      # cheapest per-km ever seen
    base   = 0                                               # honestly: per-km-only

The calibrated floor is deliberately per-km-only. A base derived from the same data as
`base = min(actual - per_km*dist)` is structurally ZERO - per_km's own argmin zeroes its
term and every other term is >= 0 by the choice of per_km - so pretending to fit one would
be a fiction. (A genuine base needs a joint fit, e.g. the supporting line of the lower
convex hull; noted as future work - a floor that is merely loose costs CPU, never
correctness.) For any observation i, `per_km * dist_i <= actual_i` because the min is taken
over a set that includes i. Scaling down by `margin < 1` keeps that true and leaves room
for a future fare cheaper than anything seen so far. If a genuinely lower fare still
appears, the existing floor-violation detector logs it and the calibration self-corrects on
the next run.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from .routing.floors import DEFAULT_FLOORS, ModeFloor, PriceFloorModel

#: Modes whose legs carry an exact precomputed fare (flights, Freerider cars). Their floors
#: must stay zero, so they are never calibrated - a per-km floor on top of an exact fare
#: would overshoot the truth.
_NEVER_CALIBRATE = frozenset({"flight", "freerider", "walk"})

#: Below this many observations, one operator's data is too thin to trust; keep the default.
MIN_SAMPLES = 8

#: Discount applied to the fitted floor for headroom against future cheaper fares. A fare
#: unobserved during calibration can undercut the fitted minimum, so this leaves a wide
#: margin; the floor-violation detector catches any residual overshoot and the next
#: calibration self-corrects from the newly logged fare.
SAFETY_MARGIN = 0.75


@dataclass(frozen=True, slots=True)
class CalibratedFloor:
    operator: str
    mode: str
    base_ore: int
    per_km_ore: int
    samples: int

    def as_row(self, now: datetime | None = None) -> dict:
        return {
            "operator": self.operator,
            "mode": self.mode,
            "base_ore": self.base_ore,
            "per_km_ore": self.per_km_ore,
            "samples": self.samples,
            "updated_at": (now or datetime.now(UTC)).isoformat(),
        }


def calibrate(
    observations, *, min_samples: int = MIN_SAMPLES, margin: float = SAFETY_MARGIN
) -> list[CalibratedFloor]:
    """Fit a safe, tight per-operator floor from `(operator, mode, distance_km, actual_ore)` rows."""
    groups: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for row in observations:
        operator = row["operator"]
        mode = row["mode"]
        distance = row["distance_km"]
        actual = row["actual_ore"]
        if not operator or mode in _NEVER_CALIBRATE or not distance or distance <= 0 or actual <= 0:
            continue
        groups[operator].append((mode, float(distance), int(actual)))

    calibrated: list[CalibratedFloor] = []
    for operator, points in groups.items():
        if len(points) < min_samples:
            continue
        mode = Counter(m for m, _, _ in points).most_common(1)[0][0]
        per_km = min(actual / dist for _, dist, actual in points)
        # No base term: min(actual - per_km*dist) over the SAME points is structurally zero
        # (per_km's argmin zeroes its own term; every other is >= 0), so computing one only
        # dressed a constant 0 up as a fit. See the module docstring for the honest story.
        per_km_ore = int(per_km * margin)
        calibrated.append(
            CalibratedFloor(
                operator=operator,
                mode=mode,
                base_ore=0,
                per_km_ore=per_km_ore,
                samples=len(points),
            )
        )
    return calibrated


def floors_from_rows(rows) -> PriceFloorModel:
    """Build a `PriceFloorModel` whose operator overrides come from the calibrated table.

    Rows are the `operator_floor` records; the mode-level defaults stay in place for
    operators (and modes) without a calibration.
    """
    overrides: dict[str, ModeFloor] = {}
    for row in rows:
        if row["mode"] in _NEVER_CALIBRATE:
            continue
        overrides[row["operator"]] = ModeFloor(
            base_ore=int(row["base_ore"]), per_km_ore=int(row["per_km_ore"])
        )
    return PriceFloorModel(DEFAULT_FLOORS, overrides)


def load_calibrated_floors(db) -> PriceFloorModel:
    """The floor model the planner should use: defaults refined by whatever calibration exists."""
    return floors_from_rows(db.get_operator_floors())


def run_calibration(db, *, min_samples: int = MIN_SAMPLES, margin: float = SAFETY_MARGIN) -> list[CalibratedFloor]:
    """Read observations, fit floors, persist them. Returns what was written."""
    calibrated = calibrate(db.reprice_observations(), min_samples=min_samples, margin=margin)
    if calibrated:
        db.put_operator_floors([c.as_row() for c in calibrated])
    return calibrated
