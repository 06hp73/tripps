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


# --- calibration base honesty ----------------------------------------------------------------


def test_calibrated_base_is_explicitly_zero():
    """min(actual - per_km*dist) over the fitting points is structurally 0 (per_km's own
    argmin zeroes its term), so the floor is per-km-only and says so."""
    obs = [
        {"operator": "OP", "mode": "train", "distance_km": 100 + i * 25,
         "actual_ore": 50_000 + i * 100}  # a genuine 500 kr intercept the fit CANNOT see
        for i in range(10)
    ]
    [floor] = calibrate(obs, min_samples=8)
    assert floor.base_ore == 0
    assert floor.per_km_ore > 0
    # The invariant still holds on every observation with the per-km-only floor.
    for o in obs:
        assert floor.per_km_ore * o["distance_km"] <= o["actual_ore"]
