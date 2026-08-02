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
the fit is provably safe on the observed data and then discounted by a margin for headroom.

The fitted floor is a LINE `base + per_km * dist` chosen among provably-safe candidates:
the through-origin line with `per_km = min(actual/dist)` (always safe: the min is over a
set including every point), plus each edge of the lower convex hull of the (dist, actual)
cloud extended to a full line (a hull edge's line supports the hull, so every observation
lies on or above it). Deriving a base the naive way - `min(actual - per_km*dist)` with
per_km already the min ratio - is structurally ZERO (the argmin zeroes its own term), which
is why a joint fit over hull edges is needed to learn a genuine boarding component at all.
Every candidate is re-verified point-by-point (belt over the geometric proof), the one with
the greatest total floor mass over the observations wins, and both coefficients are scaled
by `margin < 1` for headroom against fares cheaper than anything yet seen. If a genuinely
lower fare still appears, the floor-violation detector logs it and the next calibration
self-corrects. Note the extrapolation caveat: below the shortest observed distance a
positive base keeps the floor at `base` while a real short-hop fare could undercut it -
the same unobserved-region risk the per-km slope always had, covered by the same margin
and detector.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from .models import ADULT, Passenger, PassengerCategory
from .routing.floors import DEFAULT_FLOORS, ModeFloor, PriceFloorModel

#: Modes whose legs carry an exact precomputed fare (flights, Freerider cars). Their floors
#: must stay zero, so they are never calibrated - a per-km floor on top of an exact fare
#: would overshoot the truth.
_NEVER_CALIBRATE = frozenset({"flight", "freerider", "walk"})

#: Below this many observations, one operator's data is too thin to trust; keep the default.
MIN_SAMPLES = 8

#: What an operator's ADULT floor is multiplied by when routing for a discounted traveller
#: whose own category has not accrued `MIN_SAMPLES` yet.
#:
#: The adult floor is fitted to sit just under the cheapest adult fare seen, so a discounted
#: fare goes straight through it - and a floor above the truth can prune the genuinely
#: cheapest journey, the one failure this project refuses to ship. The deepest discount
#: measured on 2026-08-02 was 20% off (SJ and Tora student, 411/515 and 423/530; senior 0.90,
#: child 0.85), so halving leaves better than twice the observed headroom while still being
#: far tighter than the hand-set defaults. A category that accrues its own samples stops
#: using this and gets a real fit; until then the violation detector backstops it.
DISCOUNT_FLOOR_SCALE = 0.5


def _row_passenger(row) -> str:
    """The category a stored row belongs to, defaulting to adult for pre-category rows."""
    try:
        value = row["passenger"]
    except (KeyError, IndexError):
        return PassengerCategory.ADULT.value
    return value or PassengerCategory.ADULT.value

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
    passenger: str = PassengerCategory.ADULT.value

    def as_row(self, now: datetime | None = None) -> dict:
        return {
            "operator": self.operator,
            "passenger": self.passenger,
            "mode": self.mode,
            "base_ore": self.base_ore,
            "per_km_ore": self.per_km_ore,
            "samples": self.samples,
            "updated_at": (now or datetime.now(UTC)).isoformat(),
        }


def _lower_hull(points: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Lower convex hull of (distance, fare) points, left to right (monotone chain).

    Duplicate distances collapse to their cheapest fare first - only the lower envelope can
    constrain a floor.
    """
    dedup: list[tuple[float, int]] = []
    for d, a in sorted(points):
        if dedup and dedup[-1][0] == d:
            if a < dedup[-1][1]:
                dedup[-1] = (d, a)
        else:
            dedup.append((d, a))
    hull: list[tuple[float, int]] = []
    for p in dedup:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # Pop while the middle point sits on or above the chord: keep the hull convex-down.
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def _fit_floor_line(points: list[tuple[str, float, int]]) -> tuple[float, float]:
    """The safest-tight (base, per_km) line under every observation.

    Candidates: the through-origin min-ratio line (always valid), plus every lower-hull edge
    with non-negative slope and intercept (a hull edge's line supports the whole cloud).
    Each is re-verified against every point, and the one with the greatest total floor mass
    over the observations wins - the tightest bound on the traffic actually seen.
    """
    per_km_only = min(actual / dist for _, dist, actual in points)
    candidates: list[tuple[float, float]] = [(0.0, per_km_only)]
    hull = _lower_hull([(dist, actual) for _, dist, actual in points])
    for (x1, y1), (x2, y2) in zip(hull, hull[1:], strict=False):
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope >= 0 and intercept > 0:
            candidates.append((intercept, slope))
    # Epsilon absorbs float round-off at a candidate's own defining points ((a/d)*d can
    # exceed a by one ulp); 1e-6 öre is twelve orders below any fare, and the margin scaling
    # plus int() truncation keep the final integer floor strictly on the safe side.
    safe = [
        (b, s)
        for b, s in candidates
        if all(b + s * dist <= actual + 1e-6 for _, dist, actual in points)
    ]
    if not safe:  # pure paranoia: the through-origin line is valid up to that same round-off
        return (0.0, per_km_only)
    return max(safe, key=lambda c: sum(c[0] + c[1] * dist for _, dist, _ in points))


def calibrate(
    observations, *, min_samples: int = MIN_SAMPLES, margin: float = SAFETY_MARGIN
) -> list[CalibratedFloor]:
    """Fit a safe, tight per-operator floor from `(operator, mode, distance_km, actual_ore)` rows."""
    # Keyed by (operator, passenger): one operator sells the same kilometre at several prices,
    # and a fit pooling them would sit under the adult fares by the size of the discount -
    # safe but slack - while a fit of adult rows alone would sit ABOVE the student fare it
    # never saw. Each category is fitted only against fares actually quoted for it.
    groups: dict[tuple[str, str], list[tuple[str, float, int]]] = defaultdict(list)
    for row in observations:
        operator = row["operator"]
        mode = row["mode"]
        distance = row["distance_km"]
        actual = row["actual_ore"]
        if not operator or mode in _NEVER_CALIBRATE or not distance or distance <= 0 or actual <= 0:
            continue
        passenger = _row_passenger(row)
        groups[(operator, passenger)].append((mode, float(distance), int(actual)))

    calibrated: list[CalibratedFloor] = []
    for (operator, passenger), points in groups.items():
        if len(points) < min_samples:
            continue
        mode = Counter(m for m, _, _ in points).most_common(1)[0][0]
        base, per_km = _fit_floor_line(points)
        calibrated.append(
            CalibratedFloor(
                operator=operator,
                mode=mode,
                base_ore=int(base * margin),
                per_km_ore=int(per_km * margin),
                samples=len(points),
                passenger=passenger,
            )
        )
    return calibrated


def floors_from_rows(rows, passenger: Passenger = ADULT) -> PriceFloorModel:
    """Build a `PriceFloorModel` whose operator overrides come from the calibrated table.

    Rows are the `operator_floor` records; the mode-level defaults stay in place for
    operators (and modes) without a calibration.

    An operator with rows fitted for this traveller's own category uses them. One with only
    adult rows falls back to the adult floor scaled by `DISCOUNT_FLOOR_SCALE` - never the
    adult floor itself, which a discounted fare would slip under.
    """
    by_category: dict[tuple[str, str], ModeFloor] = {}
    for row in rows:
        if row["mode"] in _NEVER_CALIBRATE:
            continue
        by_category[(row["operator"], _row_passenger(row))] = ModeFloor(
            base_ore=int(row["base_ore"]), per_km_ore=int(row["per_km_ore"])
        )

    category = passenger.category.value
    adult = PassengerCategory.ADULT.value
    overrides: dict[str, ModeFloor] = {}
    for (operator, row_category), floor in by_category.items():
        if row_category == category:
            overrides[operator] = floor
        elif row_category == adult and (operator, category) not in by_category:
            overrides[operator] = ModeFloor(
                base_ore=int(floor.base_ore * DISCOUNT_FLOOR_SCALE),
                per_km_ore=int(floor.per_km_ore * DISCOUNT_FLOOR_SCALE),
            )
    return PriceFloorModel(DEFAULT_FLOORS, overrides)


def load_calibrated_floors(db, passenger: Passenger = ADULT) -> PriceFloorModel:
    """The floor model the planner should use: defaults refined by whatever calibration exists."""
    return floors_from_rows(db.get_operator_floors(), passenger)


def run_calibration(db, *, min_samples: int = MIN_SAMPLES, margin: float = SAFETY_MARGIN) -> list[CalibratedFloor]:
    """Read observations, fit floors, persist them. Returns what was written."""
    calibrated = calibrate(db.reprice_observations(), min_samples=min_samples, margin=margin)
    if calibrated:
        db.put_operator_floors([c.as_row() for c in calibrated])
    return calibrated
