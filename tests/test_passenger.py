"""Passenger categories: the traveller model, cache identity, and the floor contract.

The live half of this (that SJ really answers 411 SEK for a student where an adult pays 515)
is probed by `tripps canary` and by hand; what is pinned here is everything that can silently
go wrong *around* the discount:

* a discounted fare served to - or from - the wrong traveller's cache entry,
* a student fare filed as an adult calibration sample, dragging that operator's adult floor
  down for every later adult search,
* an adult-fitted floor applied to a discounted search, which sits ABOVE the fare and lets
  McRAPTOR prune the genuinely cheapest journey. That one is the whole project's invariant.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tripps.calibration import (
    DISCOUNT_FLOOR_SCALE,
    calibrate,
    floors_from_rows,
    run_calibration,
)
from tripps.db import Database, leg_cache_key
from tripps.models import (
    ADULT,
    Leg,
    Passenger,
    PassengerCategory,
    PriceConfidence,
    Quote,
    Stop,
    TransportMode,
)
from tripps.pricing.sj import SJ_CATEGORIES, passenger_payload
from tripps.pricing.tora import TORA_CATEGORIES
from tripps.routing.floors import ModeFloor, PriceFloorModel

TZ = ZoneInfo("Europe/Stockholm")

STHLM = Stop(id="740000001", name="Stockholm C", lat=59.330, lon=18.059)
GBG = Stop(id="740000002", name="Göteborg C", lat=57.708, lon=11.973)

STUDENT = Passenger(category=PassengerCategory.STUDENT)
CHILD = Passenger(category=PassengerCategory.CHILD, age=8)


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "p.sqlite3")
    yield database
    database.close()


def _leg() -> Leg:
    return Leg(
        from_stop=STHLM,
        to_stop=GBG,
        departure=datetime(2026, 8, 10, 5, 13, tzinfo=TZ),
        arrival=datetime(2026, 8, 10, 10, 0, tzinfo=TZ),
        mode=TransportMode.TRAIN,
        operator="SJ",
        service_ref="SJ-401",
    )


# --- the traveller ---------------------------------------------------------


def test_adult_carries_no_age_and_discounts_default_theirs():
    assert ADULT.age is None
    assert Passenger(category=PassengerCategory.STUDENT).age == 22
    assert Passenger(category=PassengerCategory.SENIOR).age == 70


def test_age_outside_the_tier_is_refused_here_not_upstream():
    """SJ answers 400 for a 26-year-old CHILD_AND_YOUTH; catching it locally names the field."""
    with pytest.raises(ValueError, match="ages 0-25"):
        Passenger(category=PassengerCategory.YOUTH, age=26)
    with pytest.raises(ValueError, match="ages 15-120"):
        Passenger(category=PassengerCategory.STUDENT, age=14)


def test_passenger_is_frozen():
    with pytest.raises(ValueError):
        STUDENT.category = PassengerCategory.ADULT


# --- cache identity --------------------------------------------------------


def test_each_traveller_gets_their_own_cache_key():
    leg = _leg()
    adult = leg_cache_key("sj", leg, ADULT)
    student = leg_cache_key("sj", leg, STUDENT)
    child = leg_cache_key("sj", leg, CHILD)
    assert len({adult, student, child}) == 3


def test_adult_key_is_unchanged_so_existing_cache_rows_survive():
    """Adult keys are written unsuffixed; fares cached before categories existed stay usable."""
    leg = _leg()
    assert leg_cache_key("sj", leg) == leg_cache_key("sj", leg, ADULT)


def test_age_is_part_of_the_key_because_it_is_part_of_the_price():
    """Child 8 and youth 16 are both CHILD_AND_YOUTH to SJ, at 437 and 411 SEK."""
    leg = _leg()
    eight = leg_cache_key("sj", leg, Passenger(category=PassengerCategory.YOUTH, age=8))
    sixteen = leg_cache_key("sj", leg, Passenger(category=PassengerCategory.YOUTH, age=16))
    assert eight != sixteen


def test_a_student_fare_is_never_served_to_an_adult(db):
    leg = _leg()
    db.put_quote(
        "sj",
        leg,
        Quote(source="sj", amount_ore=41_100, confidence=PriceConfidence.EXACT),
        600,
        STUDENT,
    )
    assert db.get_quote("sj", leg, passenger=STUDENT).amount_ore == 41_100
    assert db.get_quote("sj", leg) is None
    assert db.get_quote("sj", leg, passenger=ADULT) is None


# --- upstream payloads -----------------------------------------------------


def test_sj_puts_the_age_inside_the_category():
    """Anywhere else and SJ answers 400 'Age cannot be null for type STUDENT'."""
    payload = passenger_payload(STUDENT)
    assert payload == {"passengerCategory": {"type": "STUDENT", "age": 22}}
    assert passenger_payload(ADULT) == {"passengerCategory": {"type": "ADULT"}}


def test_sj_has_no_child_tier_so_a_child_is_a_young_child_and_youth():
    assert SJ_CATEGORIES[PassengerCategory.CHILD] == "CHILD_AND_YOUTH"
    assert SJ_CATEGORIES[PassengerCategory.YOUTH] == "CHILD_AND_YOUTH"
    assert passenger_payload(CHILD)["passengerCategory"]["age"] == 8


def test_sources_declare_only_the_tiers_they_really_sell():
    """Tora 400s on YOUTH and CHILD, and FlixBus Sweden sells no discounted ticket at all."""
    from tripps.pricing.flixbus import FlixBusAdapter
    from tripps.pricing.tora import ToraAdapter

    assert set(TORA_CATEGORIES) == {
        PassengerCategory.ADULT,
        PassengerCategory.STUDENT,
        PassengerCategory.SENIOR,
    }
    tora = ToraAdapter(min_interval=0.0)
    assert tora.prices_natively(STUDENT)
    assert not tora.prices_natively(Passenger(category=PassengerCategory.YOUTH))

    flix = FlixBusAdapter(min_interval=0.0)
    assert flix.prices_natively(ADULT)
    assert not flix.prices_natively(STUDENT)
    assert "adult price shown" in flix.category_fallback_note(STUDENT)
    assert flix.category_fallback_note(ADULT) is None


# --- calibration -----------------------------------------------------------


def _rows(passenger: str, fares: list[tuple[float, int]]) -> list[dict]:
    return [
        {
            "operator": "SJ",
            "mode": "train",
            "distance_km": dist,
            "actual_ore": ore,
            "passenger": passenger,
        }
        for dist, ore in fares
    ]


def test_each_category_is_fitted_against_its_own_fares_only():
    adult = _rows("adult", [(100 + i, 50_000 + i * 100) for i in range(10)])
    student = _rows("student", [(100 + i, 40_000 + i * 100) for i in range(10)])
    fitted = {(f.operator, f.passenger): f for f in calibrate(adult + student)}

    assert set(fitted) == {("SJ", "adult"), ("SJ", "student")}
    assert fitted[("SJ", "adult")].samples == 10
    # The student fit must sit under the student fares, which the adult fit would not.
    cheapest_student = 40_000
    s = fitted[("SJ", "student")]
    assert s.base_ore + s.per_km_ore * 100 <= cheapest_student


def test_a_student_fare_never_lands_in_the_adult_fit(db):
    for i in range(10):
        db.record_reprice_delta(
            source="sj", mode="train", operator="SJ",
            distance_km=400 + i, floor_ore=10_000, actual_ore=41_100 + i, passenger=STUDENT,
        )
    calibrated = run_calibration(db)
    assert [c.passenger for c in calibrated] == ["student"]
    assert not db.get_operator_floors("adult")
    assert len(db.get_operator_floors("student")) == 1


def test_pre_category_rows_read_as_adult(db):
    """Rows written before the column existed are adult observations, and stay usable."""
    with db._write() as conn:
        conn.execute(
            "INSERT INTO reprice_delta "
            "(recorded_at, source, mode, operator, distance_km, floor_ore, actual_ore) "
            "VALUES ('2026-07-01T00:00:00+00:00', 'sj', 'train', 'SJ', 400, 1000, 50000)"
        )
    assert db.reprice_observations()[0]["passenger"] == "adult"


# --- the floor contract ----------------------------------------------------


def test_a_discounted_search_never_uses_the_bare_adult_floor():
    """The adult floor sits just under adult fares, so a student fare slips beneath it."""
    rows = [
        {"operator": "SJ", "passenger": "adult", "mode": "train",
         "base_ore": 40_000, "per_km_ore": 20, "samples": 12}
    ]
    adult_model = floors_from_rows(rows, ADULT)
    student_model = floors_from_rows(rows, STUDENT)

    adult_floor = adult_model.floor_ore(TransportMode.TRAIN, "SJ", 400)
    student_floor = student_model.floor_ore(TransportMode.TRAIN, "SJ", 400)
    assert student_floor < adult_floor
    assert student_floor == int(
        int(40_000 * DISCOUNT_FLOOR_SCALE) + int(20 * DISCOUNT_FLOOR_SCALE) * 400
    )
    # The real observation this protects: SJ 411 SEK student against a 515 SEK adult fare.
    assert student_floor <= 41_100


def test_a_category_with_its_own_fit_uses_it_rather_than_the_scaled_adult():
    rows = [
        {"operator": "SJ", "passenger": "adult", "mode": "train",
         "base_ore": 40_000, "per_km_ore": 20, "samples": 12},
        {"operator": "SJ", "passenger": "student", "mode": "train",
         "base_ore": 30_000, "per_km_ore": 15, "samples": 12},
    ]
    model = floors_from_rows(rows, STUDENT)
    assert model.floor_ore(TransportMode.TRAIN, "SJ", 0) == 30_000


def test_scaling_a_floor_up_is_refused():
    """Downward is always safe; upward is exactly how the cheapest journey gets pruned."""
    model = PriceFloorModel({TransportMode.TRAIN: ModeFloor(10_000, 50)})
    assert model.scaled(0.5).floor_ore(TransportMode.TRAIN, None, 100) == 10_000 // 2 + 25 * 100
    with pytest.raises(ValueError, match="can break floor"):
        model.scaled(1.5)


def test_scaled_floors_stay_under_every_observed_discount():
    """0.5 leaves better than twice the headroom of the deepest discount measured (0.80)."""
    deepest_observed_ratio = 411 / 515
    assert DISCOUNT_FLOOR_SCALE < deepest_observed_ratio
