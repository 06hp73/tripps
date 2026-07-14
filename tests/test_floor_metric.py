"""One distance metric for the price floor: fit, application, and audit all use the ridden path.

The bug class under test: McRAPTOR integrates `per_km` over the route's per-segment polyline,
while calibration used to fit (and the violation detector used to audit) the endpoint
straight-line. Polyline >= chord, so a per-km rate fit on the chord and applied over the curve
exceeded the very fare it was fit from on winding routes (Öresundståg via the bridge: 2.68x) -
and the detector, measuring the chord too, was blind exactly there.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from tripps.calibration import calibrate
from tripps.config import PricingBudget
from tripps.db import Database
from tripps.models import Leg, PriceConfidence, Quote, Stop, TransportMode
from tripps.pricing.base import CallBudget
from tripps.pricing.orchestrator import PricingOrchestrator, _Context
from tripps.routing.floors import ModeFloor, PriceFloorModel
from tripps.routing.journey import _ride_leg, leg_distance_km, leg_floor_km
from tripps.routing.mcraptor import RideEdge
from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip

# An L-shaped line: A -> B is ~111 km north, B -> C is ~111 km east, so the ridden path is
# ~222 km while the A -> C chord is ~157 km (ratio ~1.41 - past the 1/0.75 breach threshold).
A = Stop(id="A", name="A", lat=55.0, lon=13.0)
B = Stop(id="B", name="B", lat=56.0, lon=13.0)
C = Stop(id="C", name="C", lat=56.0, lon=14.8)


def _l_shaped_timetable():
    b = TimetableBuilder()
    for s in (A, B, C):
        b.add_stop(s)
    b.add_trip(
        RouteInfo(id="L", mode=TransportMode.TRAIN, operator="Winding"),
        ["A", "B", "C"],
        Trip(id="L1", arrivals=[0, 3600, 7200], departures=[0, 3600, 7200]),
    )
    return b.build()


def _leg(path_km: float | None) -> Leg:
    return Leg(
        from_stop=A,
        to_stop=C,
        mode=TransportMode.TRAIN,
        operator="Winding",
        departure=datetime(2026, 8, 3, 8, 0),
        arrival=datetime(2026, 8, 3, 10, 0),
        service_ref="L1",
        path_km=path_km,
    )


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "m.sqlite3")
    yield database
    database.close()


# --- the leg carries the router's own metric ------------------------------------------------


def test_ride_leg_carries_the_routers_polyline_distance():
    tt = _l_shaped_timetable()
    edge = RideEdge(route_idx=0, trip_idx=0, board_pos=0, alight_pos=2)
    leg = _ride_leg(tt, edge, date(2026, 8, 3))
    assert leg.path_km == pytest.approx(sum(tt.routes[0].segment_km))
    # The polyline is genuinely longer than the chord on this shape - the whole point.
    assert leg.path_km > leg_distance_km(leg) * 1.34


def test_leg_floor_km_prefers_path_and_falls_back_to_chord():
    assert leg_floor_km(_leg(path_km=222.0)) == 222.0
    fallback = _leg(path_km=None)
    assert leg_floor_km(fallback) == pytest.approx(leg_distance_km(fallback))


# --- fit-on-path keeps the invariant where fit-on-chord broke it ----------------------------


def test_fit_and_apply_on_the_same_path_metric_keeps_floor_under_fare():
    """The audit's numeric repro. Fare 224 SEK, chord 233 km, path 326 km (ratio 1.40).

    Old behavior: per_km fit on the chord = int(22400/233 * 0.75) = 72, applied by the router
    over the path = 72 * 326 = 23472 öre > 22400 - invariant broken, undetectably. New
    behavior: the sample is recorded with the path distance, so the fit itself guarantees
    per_km * path <= actual for the observing leg.
    """
    path = 326.0
    obs = [
        {"operator": "Kust", "mode": "train", "distance_km": path, "actual_ore": 22_400}
        for _ in range(8)
    ]
    [floor] = calibrate(obs, min_samples=8)
    assert floor.base_ore + int(floor.per_km_ore * path) <= 22_400
    # And the chord-fit arithmetic really would have breached (the bug we are killing):
    chord_per_km = int((22_400 / 233.0) * 0.75)
    assert chord_per_km * path > 22_400


# --- the orchestrator records and audits the path metric ------------------------------------


def _orch(db, floors=None):
    return PricingOrchestrator(
        [], db, budget=PricingBudget(min_interval_seconds=0.0), floors=floors or PriceFloorModel()
    )


def test_record_delta_logs_the_path_distance(db):
    orch = _orch(db)
    quote = Quote(source="stub", amount_ore=22_400, confidence=PriceConfidence.EXACT)
    orch._record_delta(_leg(path_km=326.0), quote, "stub")
    [row] = db.reprice_observations()
    assert row["distance_km"] == pytest.approx(326.0)


def test_check_floor_audits_the_path_so_the_detector_is_no_longer_blind(db):
    """per_km=72 over path 326 = 23472 > 22400: a real violation. The old chord audit
    (72 * 233 = 16776 <= 22400) reported nothing - blind exactly where the floor broke."""
    floors = PriceFloorModel(operator_overrides={"Winding": ModeFloor(base_ore=0, per_km_ore=72)})
    orch = _orch(db, floors=floors)
    ctx = _Context(call_budget=CallBudget.from_settings(PricingBudget()))
    quote = Quote(source="stub", amount_ore=22_400, confidence=PriceConfidence.EXACT)
    orch._check_floor(_leg(path_km=326.0), quote, ctx)
    assert len(ctx.violations) == 1


# --- one-time migration wipes chord-metric data ----------------------------------------------


def test_metric_migration_wipes_legacy_rows_and_keeps_current_ones(tmp_path):
    path = tmp_path / "mig.sqlite3"

    first = Database(path)  # fresh DB: marker set to 'path'
    first.record_reprice_delta(
        source="t", mode="train", operator="OP", distance_km=100.0, floor_ore=1, actual_ore=9000
    )
    first.put_operator_floors(
        [{"operator": "OP", "mode": "train", "base_ore": 0, "per_km_ore": 5,
          "samples": 8, "updated_at": "2026-07-14T00:00:00+00:00"}]
    )
    first.close()

    # Reopen under the SAME metric: nothing is wiped.
    second = Database(path)
    assert len(second.reprice_observations()) == 1
    assert len(second.get_operator_floors()) == 1
    # Forge a legacy marker, as if these rows had been recorded under the chord metric.
    with second._write() as conn:
        conn.execute("UPDATE meta SET value = 'chord' WHERE key = 'floor_distance_metric'")
    second.close()

    third = Database(path)  # metric mismatch: both tables wiped, marker restored
    assert third.reprice_observations() == []
    assert third.get_operator_floors() == []
    row = third._conn.execute(
        "SELECT value FROM meta WHERE key = 'floor_distance_metric'"
    ).fetchone()
    assert row["value"] == "path"
    third.close()
