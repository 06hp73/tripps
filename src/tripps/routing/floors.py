"""Price lower bounds used as a McRAPTOR search criterion.

These numbers are intentionally *pessimistic* (too low), not accurate. The contract in
`interfaces.PriceFloor` is one-sided:

    floor(leg) <= true_price(leg)

Violating it can prune the genuinely cheapest itinerary before phase 2 ever prices it,
which is a silent wrong answer. Being too far below the truth merely widens the Pareto
frontier and costs a little CPU. So every default here sits under the cheapest promo
fare we have ever observed, and calibration is expected to *raise* them later from
recorded `reprice_delta` rows rather than to lower them.

Injected modes (FLIGHT, FREERIDER) carry an exact `precomputed_fare_ore` on the trip
itself, so their floor components are pinned to zero: adding a per-km bound on top of an
already-exact fare would overshoot the truth and break the contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..interfaces import PriceFloor
from ..models import SEK, TransportMode


@dataclass(frozen=True, slots=True)
class ModeFloor:
    """Lower bounds for one mode: once per boarding, plus per kilometre ridden."""

    base_ore: int
    per_km_ore: int

    def __post_init__(self) -> None:
        if self.base_ore < 0 or self.per_km_ore < 0:
            raise ValueError("price floors cannot be negative")


#: Modes whose trips always carry an exact fare, so the floor must contribute nothing.
EXACT_FARE_MODES = frozenset({TransportMode.FLIGHT, TransportMode.FREERIDER})

#: Observed anchors that these bounds must stay under (2026-07-10, verified live):
#:   FlixBus Stockholm -> Goteborg, ~470 road km, cheapest 408 SEK  => 0.87 SEK/km.
#: FlixBus runs 99 SEK promos on comparable distances, i.e. ~0.21 SEK/km, so the bus
#: per-km bound is set an order of magnitude below even that.
DEFAULT_FLOORS: dict[TransportMode, ModeFloor] = {
    # SJ's cheapest advance "Ja tack"-class tickets start around 95 SEK; regional
    # single tickets can be far lower, so the base is put below any of them.
    TransportMode.TRAIN: ModeFloor(base_ore=19 * SEK, per_km_ore=5),
    TransportMode.BUS: ModeFloor(base_ore=9 * SEK, per_km_ore=2),
    TransportMode.FERRY: ModeFloor(base_ore=0, per_km_ore=0),
    # Schedules exist, no price source. Priced as an estimate in phase 2 and flagged;
    # contributing 0 here keeps it from distorting the routing objective.
    TransportMode.LOCAL_TRANSIT: ModeFloor(base_ore=0, per_km_ore=0),
    TransportMode.FLIGHT: ModeFloor(base_ore=0, per_km_ore=0),
    TransportMode.FREERIDER: ModeFloor(base_ore=0, per_km_ore=0),
    TransportMode.WALK: ModeFloor(base_ore=0, per_km_ore=0),
}


class PriceFloorModel(PriceFloor):
    """Table-driven floors with optional per-operator refinement."""

    def __init__(
        self,
        table: dict[TransportMode, ModeFloor] | None = None,
        operator_overrides: dict[str, ModeFloor] | None = None,
    ) -> None:
        self._table = dict(table or DEFAULT_FLOORS)
        self._operators = dict(operator_overrides or {})
        for mode in EXACT_FARE_MODES:
            floor = self._table.get(mode)
            if floor is not None and (floor.base_ore or floor.per_km_ore):
                raise ValueError(
                    f"{mode.value} trips carry exact fares; its floor must be zero, got {floor}"
                )

    def _lookup(self, mode: TransportMode, operator: str | None) -> ModeFloor:
        if operator is not None:
            override = self._operators.get(operator)
            if override is not None:
                return override
        return self._table.get(mode, ModeFloor(0, 0))

    def boarding_ore(self, mode: TransportMode, operator: str | None) -> int:
        return self._lookup(mode, operator).base_ore

    def per_km_ore(self, mode: TransportMode, operator: str | None) -> int:
        return self._lookup(mode, operator).per_km_ore

    def with_operator(self, operator: str, floor: ModeFloor) -> PriceFloorModel:
        return PriceFloorModel(self._table, {**self._operators, operator: floor})

    def scaled(self, factor: float) -> PriceFloorModel:
        """Every bound multiplied by `factor`, for pricing a traveller who pays less.

        Only safe downward, and that is the only direction it is used: for `factor <= 1` the
        result stays at or below the original, and the original is already at or below every
        adult fare, so the contract survives any discount at least as deep as the factor.
        Truncation to int rounds toward zero, which is also the safe side.
        """
        if factor > 1:
            raise ValueError(f"scaling a floor UP can break floor <= true price: {factor}")
        table = {
            mode: ModeFloor(int(f.base_ore * factor), int(f.per_km_ore * factor))
            for mode, f in self._table.items()
        }
        operators = {
            name: ModeFloor(int(f.base_ore * factor), int(f.per_km_ore * factor))
            for name, f in self._operators.items()
        }
        return PriceFloorModel(table, operators)

    def with_operators_zeroed(self, operators) -> PriceFloorModel:
        """A copy where the given operators have a zero floor (for pass-aware routing).

        Zero is a valid lower bound for any leg, so the floor contract (floor <= true price)
        is preserved even for an operator whose legs are only sometimes covered.
        """
        overrides = {**self._operators, **{op: ModeFloor(0, 0) for op in operators}}
        return PriceFloorModel(self._table, overrides)


def zero_floors() -> PriceFloorModel:
    """All bounds zero: routing degenerates to earliest-arrival plus exact injected fares.

    Useful as a correctness oracle in tests. Any journey the calibrated model finds must
    also be findable here, because zero is a valid (if useless) lower bound.
    """
    return PriceFloorModel({mode: ModeFloor(0, 0) for mode in TransportMode})
