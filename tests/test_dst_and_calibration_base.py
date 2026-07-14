"""Two honesty fixes: max_duration enforces TRUE elapsed time, and the calibrated floor
admits it is per-km-only (a base fit from the same min is structurally zero)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tripps.calibration import calibrate
from tripps.models import Itinerary, Leg, SearchConstraints, Stop, TransportMode

TZ = ZoneInfo("Europe/Stockholm")
A = Stop(id="A", name="A", lat=59.33, lon=18.06)
B = Stop(id="B", name="B", lat=57.71, lon=11.97)


def _overnight_fallback_itinerary() -> Itinerary:
    """Departs 01:30 CEST on the 2026-10-25 fall-back night, arrives 03:30 CET: the clock
    face shows 2 h, the passenger sits on the train for 3 h (02:00-03:00 repeats)."""
    dep = datetime(2026, 10, 25, 1, 30, tzinfo=TZ)
    arr = datetime(2026, 10, 25, 3, 30, tzinfo=TZ)
    return Itinerary(
        legs=[
            Leg(
                from_stop=A, to_stop=B, mode=TransportMode.TRAIN, operator="SJ",
                departure=dep, arrival=arr, service_ref="night",
            )
        ]
    )


def test_displayed_duration_is_wall_nominal_by_design():
    assert _overnight_fallback_itinerary().duration_seconds == 7200


def test_max_duration_enforces_true_elapsed_across_fall_back():
    """A 9000 s (2.5 h) cap must reject the 3 h-real journey even though the clock face
    reads 2 h - otherwise the constraint quietly lies on every autumn transition night."""
    itin = _overnight_fallback_itinerary()
    assert not SearchConstraints(max_duration_seconds=9000).permits(itin)
    assert SearchConstraints(max_duration_seconds=11000).permits(itin)


def test_max_duration_on_naive_datetimes_uses_wall_difference():
    dep = datetime(2026, 7, 22, 8, 0)
    arr = datetime(2026, 7, 22, 10, 0)
    itin = Itinerary(
        legs=[Leg(from_stop=A, to_stop=B, mode=TransportMode.TRAIN, operator="SJ",
                  departure=dep, arrival=arr, service_ref="t")]
    )
    assert SearchConstraints(max_duration_seconds=7200).permits(itin)
    assert not SearchConstraints(max_duration_seconds=7199).permits(itin)


# --- calibration hull fit ---------------------------------------------------------------------


def test_hull_fit_recovers_a_genuine_base_component():
    """Fares on the exact line 496 kr + 0.04 kr/km: the old min-ratio-then-residual fit was
    structurally base=0 (its per_km absorbed the intercept as slope), giving a floor of only
    ~375 kr at 100 km. The hull fit learns the true line - base 372 kr after margin - a far
    tighter floor at every observed distance, still under every fare."""
    obs = [
        {"operator": "OP", "mode": "train", "distance_km": 100 + i * 25,
         "actual_ore": 50_000 + i * 100}  # = 49_600 + 4 * dist
        for i in range(10)
    ]
    [floor] = calibrate(obs, min_samples=8)
    assert floor.base_ore == int(49_600 * 0.75)  # 37_200: the genuine intercept, learned
    assert floor.per_km_ore == int(4 * 0.75)
    for o in obs:
        assert floor.base_ore + floor.per_km_ore * o["distance_km"] <= o["actual_ore"]
    # And it strictly beats the old per-km-only bound on the shortest observed leg.
    old_per_km_only = int(min(o["actual_ore"] / o["distance_km"] for o in obs) * 0.75)
    assert floor.base_ore + floor.per_km_ore * 100 > old_per_km_only * 100
