"""Floor calibration: the safety invariant (floor <= every observed fare) above all else."""

from __future__ import annotations

from pathlib import Path

import pytest

from tripps.calibration import (
    calibrate,
    floors_from_rows,
    load_calibrated_floors,
    run_calibration,
)
from tripps.db import Database
from tripps.models import TransportMode


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "c.sqlite3")
    yield database
    database.close()


def _obs(operator, mode, distance_km, actual_ore):
    return {"operator": operator, "mode": mode, "distance_km": distance_km, "actual_ore": actual_ore}


def _seed_delta(db, operator, mode, distance_km, actual_ore):
    db.record_reprice_delta(
        source="test", mode=mode, operator=operator, distance_km=distance_km,
        floor_ore=1, actual_ore=actual_ore,
    )


# --- the invariant ---------------------------------------------------------


def test_calibrated_floor_never_exceeds_any_observed_fare():
    """The whole correctness contract: floor(distance) <= actual for every observation."""
    obs = [_obs("SJ", "train", 100 + i * 30, 20_000 + i * 900) for i in range(12)]
    # add a couple of cheap outliers
    obs.append(_obs("SJ", "train", 400, 9_500))
    obs.append(_obs("SJ", "train", 250, 12_000))

    [floor] = calibrate(obs, min_samples=8)
    for o in obs:
        predicted = floor.base_ore + int(floor.per_km_ore * o["distance_km"])
        assert predicted <= o["actual_ore"], (o, predicted)


def test_margin_keeps_the_floor_strictly_under_the_tightest_fit():
    obs = [_obs("OP", "bus", 100 + i * 10, 10_000 + i * 100) for i in range(10)]
    loose = calibrate(obs, min_samples=8, margin=1.0)[0]
    tight = calibrate(obs, min_samples=8, margin=0.9)[0]
    assert tight.per_km_ore <= loose.per_km_ore
    assert tight.base_ore <= loose.base_ore


def test_calibration_is_tighter_than_the_conservative_default():
    """The point of the exercise: a fitted floor beats the hand-set default."""
    from tripps.routing.floors import DEFAULT_FLOORS

    obs = [_obs("SJ", "train", 450, 40_000 + i * 500) for i in range(15)]
    [floor] = calibrate(obs, min_samples=8)
    default = DEFAULT_FLOORS[TransportMode.TRAIN]
    fitted = floor.base_ore + int(floor.per_km_ore * 450)
    default_at_450 = default.base_ore + int(default.per_km_ore * 450)
    assert fitted > default_at_450, "calibrated floor should be tighter than the default"


# --- gating ----------------------------------------------------------------


def test_thin_operators_are_not_calibrated():
    obs = [_obs("Rare", "train", 100, 20_000) for _ in range(3)]
    assert calibrate(obs, min_samples=8) == []


def test_exact_fare_modes_are_never_calibrated():
    """Flights and Freerider carry an exact fare; a per-km floor on top would overshoot."""
    obs = (
        [_obs("air:SAS", "flight", 400, 80_000) for _ in range(10)]
        + [_obs("hertz-freerider", "freerider", 300, 0) for _ in range(10)]
    )
    assert calibrate(obs, min_samples=8) == []


def test_zero_and_missing_distance_rows_are_ignored():
    obs = [_obs("OP", "train", 0, 5_000), _obs("OP", "train", None, 5_000)]
    obs += [_obs("OP", "train", 100, 10_000) for _ in range(8)]
    result = calibrate(obs, min_samples=8)
    # Only the 8 valid points count toward the sample threshold.
    assert result and result[0].samples == 8


# --- persistence + loading -------------------------------------------------


def test_floors_from_rows_builds_operator_overrides():
    rows = [
        {"operator": "SJ", "mode": "train", "base_ore": 5_000, "per_km_ore": 30, "samples": 20},
        {"operator": "air:SAS", "mode": "flight", "base_ore": 9_999, "per_km_ore": 99, "samples": 20},
    ]
    model = floors_from_rows(rows)
    # SJ override applied.
    assert model.boarding_ore(TransportMode.TRAIN, "SJ") == 5_000
    assert model.per_km_ore(TransportMode.TRAIN, "SJ") == 30
    # The flight row is dropped, so flight stays at the zero default.
    assert model.boarding_ore(TransportMode.FLIGHT, "air:SAS") == 0
    assert model.per_km_ore(TransportMode.FLIGHT, "air:SAS") == 0


def test_run_calibration_reads_delta_and_persists(db):
    for i in range(12):
        _seed_delta(db, "Öresundståg", "train", 100 + i * 10, 12_900 + i * 50)

    calibrated = run_calibration(db, min_samples=8)
    assert len(calibrated) == 1
    assert calibrated[0].operator == "Öresundståg"

    # Loading builds a model that applies the persisted override.
    model = load_calibrated_floors(db)
    assert model.per_km_ore(TransportMode.TRAIN, "Öresundståg") >= 0
    floor_130km = model.floor_ore(TransportMode.TRAIN, "Öresundståg", 130)
    assert floor_130km <= 12_900, "must not exceed the cheapest observed fare"


def test_load_with_no_calibration_returns_defaults(db):
    model = load_calibrated_floors(db)
    from tripps.routing.floors import DEFAULT_FLOORS

    assert model.boarding_ore(TransportMode.TRAIN, "SJ") == DEFAULT_FLOORS[TransportMode.TRAIN].base_ore


def test_empty_observations_writes_nothing(db):
    assert run_calibration(db) == []
    assert db.get_operator_floors() == []


def test_put_operator_floors_is_a_true_replace(db):
    """An operator absent from a later batch (samples purged / below MIN_SAMPLES) must fall
    back to the conservative defaults, not keep a stale fit forever - a lingering over-tight
    floor is exactly what the calibration loop exists to prevent."""
    row = {"mode": "train", "base_ore": 0, "samples": 9, "updated_at": "2026-07-14T00:00:00+00:00"}
    db.put_operator_floors([
        {**row, "operator": "OldOp", "per_km_ore": 40},
        {**row, "operator": "KeptOp", "per_km_ore": 50},
    ])
    db.put_operator_floors([{**row, "operator": "KeptOp", "per_km_ore": 55}])

    floors = {r["operator"]: r["per_km_ore"] for r in db.get_operator_floors()}
    assert floors == {"KeptOp": 55}, "OldOp's stale fit must be gone"


# --- hull fit adversarial cases ---------------------------------------------


def test_hull_fit_never_exceeds_any_fare_with_a_cheap_short_promo():
    """A short-distance promo outlier must cap the base: the hull line through it keeps the
    invariant, where a naive intercept fit would sail over it."""
    obs = [_obs("OP", "train", 100 + i * 40, 40_000 + i * 500) for i in range(9)]
    obs.append(_obs("OP", "train", 60, 4_900))  # 49 kr promo on a short hop
    [floor] = calibrate(obs, min_samples=8, margin=1.0)  # margin 1: the fit itself must hold
    for o in obs:
        assert floor.base_ore + floor.per_km_ore * o["distance_km"] <= o["actual_ore"] + 1


def test_hull_fit_single_distance_degenerates_to_per_km_only():
    """All observations at one distance cannot separate base from slope; the through-origin
    line is the only safe shape."""
    obs = [_obs("OP", "bus", 250.0, 20_000 + i * 300) for i in range(8)]
    [floor] = calibrate(obs, min_samples=8, margin=1.0)
    assert floor.base_ore == 0
    assert floor.per_km_ore == int(20_000 / 250.0)


def test_hull_fit_is_at_least_as_tight_as_the_old_per_km_only_fit():
    """On any data, the chosen line's total floor mass >= the through-origin candidate's,
    because that candidate is always in the pool."""
    obs = [_obs("OP", "train", 80 + i * 37, 15_000 + (i * 977) % 9_000) for i in range(12)]
    [floor] = calibrate(obs, min_samples=8, margin=1.0)
    per_km_only = min(o["actual_ore"] / o["distance_km"] for o in obs)
    new_mass = sum(floor.base_ore + floor.per_km_ore * o["distance_km"] for o in obs)
    old_mass = sum(int(per_km_only) * o["distance_km"] for o in obs)
    assert new_mass >= old_mass - len(obs)  # integer truncation slack only


def test_hull_fit_fuzz_invariant_holds_on_random_clouds():
    """200 seeded random fare clouds: the fitted line (margin=1, no headroom) must sit on or
    under EVERY observation - the cardinal floor invariant, checked exhaustively."""
    import random

    rng = random.Random(20260714)
    for _ in range(200):
        n = rng.randint(8, 25)
        obs = [
            _obs(
                "OP", "train",
                round(rng.uniform(10, 600), 1),
                rng.randint(1_500, 120_000),
            )
            for _ in range(n)
        ]
        [floor] = calibrate(obs, min_samples=8, margin=1.0)
        for o in obs:
            predicted = floor.base_ore + floor.per_km_ore * o["distance_km"]
            assert predicted <= o["actual_ore"] + 1, (floor, o)
