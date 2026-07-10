"""Freerider: schema contract against a recorded live snapshot, cost model, synthetic trips.

The fixture is a real capture of `GET /api/transport-routes/?country=SWEDEN` taken on
2026-07-10. These tests are the tripwire for the undocumented schema changing under us.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tripps.ingest.freerider import (
    EXPECTED_ALLOWANCE_RATIO,
    OPTIMISTIC_FUEL,
    TYPICAL_FUEL,
    FreeriderCostModel,
    FuelParams,
    offers_to_log_rows,
    only_within,
    parse_offers,
    schema_drift,
)
from tripps.models import SEK, TransportMode
from tripps.routing.mcraptor import RaptorQuery, run_mcraptor
from tripps.routing.synthetic import freerider_additions, freerider_route_addition
from tripps.routing.timetable import Timetable, overlay_timetable
from tripps.timeutil import to_service_seconds

TZ = ZoneInfo("Europe/Stockholm")
FIXTURE = Path(__file__).parent / "fixtures" / "freerider_sweden.json"


@pytest.fixture
def raw() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def offers(raw):
    return parse_offers(raw)


def _local(y, m, d, hh=0, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=TZ)


# --- schema contract ------------------------------------------------------


def test_fixture_parses_every_route(raw, offers):
    expected = sum(len(g["routes"]) for g in raw)
    assert len(offers) == expected == 8


def test_offer_fields_are_populated(offers):
    o = next(o for o in offers if o.pickup.city == "Uppsala")
    assert o.dropoff.city.startswith("Stockholm")
    assert o.direct_km == pytest.approx(33.16)
    assert o.included_km == 40.0
    # travelTime (27) is the allowance; originalTravelTime (23) is the actual drive.
    assert o.direct_minutes == 23
    assert o.allowed_minutes == 27
    assert o.car_model
    assert o.pickup.trac_code and o.dropoff.trac_code
    assert -90 <= o.pickup.lat <= 90


def test_timestamps_are_stockholm_local_not_utc(offers):
    """The endpoint emits naive ISO strings; reading them as UTC would shift every
    pickup by one or two hours and silently break window feasibility."""
    o = offers[0]
    assert o.available_at.tzinfo is not None
    assert o.available_at.utcoffset() in (timedelta(hours=1), timedelta(hours=2))


def test_distance_is_the_mileage_allowance_not_the_drive(offers):
    """`distance` == `originalDistance` * 1.2 on every live route. The cost model
    depends on this reading; if it drifts, the model is wrong."""
    assert schema_drift(offers) == []
    for o in offers:
        assert o.allowance_ratio() == pytest.approx(EXPECTED_ALLOWANCE_RATIO, abs=0.05)
        assert o.included_km > o.direct_km


def test_schema_drift_detects_changed_allowance_semantics(offers):
    import dataclasses

    tampered = [dataclasses.replace(offers[0], included_km=offers[0].direct_km)]
    problems = schema_drift(tampered)
    assert len(problems) == 1 and "expected" in problems[0]


def test_cross_border_offers_are_excluded(offers):
    """`?country=SWEDEN` selects by destination: a Kirkenes (NO) -> Stockholm car is in
    the feed. This planner routes strictly within Sweden."""
    assert any(o.pickup.country == "no" for o in offers), "fixture must cover this case"
    swedish = only_within(offers, "se")
    assert all(o.pickup.country == "se" and o.dropoff.country == "se" for o in swedish)
    assert len(swedish) == len(offers) - 1


def test_log_rows_roundtrip(raw, offers):
    rows = offers_to_log_rows(offers, raw)
    assert len(rows) == len(offers)
    assert all(r["payload"] for r in rows), "each row keeps its raw payload"
    assert {r["route_id"] for r in rows} == {o.route_id for o in offers}


# --- cost model -----------------------------------------------------------


def test_short_offer_is_free(offers):
    """33 km on a free first tank costs nothing. This is why Freerider legs are the
    cheapest edge in the whole network."""
    model = FreeriderCostModel()
    uppsala = next(o for o in offers if o.pickup.city == "Uppsala")
    assert model.estimate_ore(uppsala) == 0
    assert model.floor_ore(uppsala) == 0
    assert "free" in model.explain(uppsala).lower()


def test_long_offer_costs_fuel_beyond_the_first_tank(offers):
    """1580 km cannot be free: the tank runs out and the driver buys fuel."""
    model = FreeriderCostModel()
    longest = max(offers, key=lambda o: o.direct_km)
    assert longest.direct_km > 1000
    assert model.estimate_ore(longest) > 0
    assert "assumption" in model.explain(longest).lower()


def test_floor_never_exceeds_estimate(offers):
    """The routing floor must sit at or below the human-facing estimate for every offer.
    A floor above the truth can prune the cheapest itinerary before it is priced."""
    model = FreeriderCostModel()
    for o in offers:
        assert model.floor_ore(o) <= model.estimate_ore(o), o.route_id


def test_overmil_only_applies_beyond_the_included_allowance(offers):
    model = FreeriderCostModel()
    o = next(o for o in offers if o.pickup.city == "Uppsala")
    within = model.estimate_ore(o, driven_km=o.included_km)
    beyond = model.estimate_ore(o, driven_km=o.included_km + 100)
    assert within == 0
    assert beyond == 100 * model.overmil_ore_per_km


def test_fuel_params_are_linear_beyond_the_tank():
    params = FuelParams(tank_range_km=500, consumption_l_per_100km=10.0, fuel_price_ore_per_litre=2000)
    assert params.fuel_cost_ore(400) == 0
    assert params.fuel_cost_ore(500) == 0
    # 100 km beyond, 10 L/100km -> 10 L at 20 SEK -> 200 SEK
    assert params.fuel_cost_ore(600) == 200 * SEK


def test_optimistic_params_are_never_pricier_than_typical():
    for km in (0, 300, 600, 900, 1600):
        assert OPTIMISTIC_FUEL.fuel_cost_ore(km) <= TYPICAL_FUEL.fuel_cost_ore(km)


# --- synthetic trips ------------------------------------------------------


def test_window_is_discretized_into_evenly_spaced_trips(offers):
    o = next(o for o in offers if o.pickup.city == "Uppsala")
    service_date = o.available_at.date()
    now = o.available_at - timedelta(hours=1)

    addition = freerider_route_addition(
        o, service_date, FreeriderCostModel(), now=now, step_minutes=30
    )
    assert addition is not None
    assert addition.info.mode is TransportMode.FREERIDER
    assert addition.info.synthetic
    assert len(addition.stops) == 2
    assert len(addition.trips) > 1

    departures = [t.departures[0] for t in addition.trips]
    assert departures == sorted(departures)
    gaps = {b - a for a, b in zip(departures, departures[1:], strict=False)}
    assert gaps == {1800}


def test_every_synthetic_trip_has_the_same_duration(offers):
    """Equal durations mean no trip overtakes another, which is precisely RAPTOR's
    route-scan precondition. Freerider therefore needs no algorithm change."""
    o = next(o for o in offers if o.pickup.city == "Uppsala")
    addition = freerider_route_addition(
        o, o.available_at.date(), FreeriderCostModel(), now=o.available_at
    )
    durations = {t.arrivals[1] - t.departures[0] for t in addition.trips}
    assert durations == {o.drive_seconds}


def test_trips_carry_the_floor_fare_not_the_estimate(offers):
    model = FreeriderCostModel()
    o = max(offers, key=lambda x: x.direct_km)  # long enough that floor != 0
    addition = freerider_route_addition(
        o, o.available_at.date(), model, now=o.available_at
    )
    assert addition is not None
    fares = {t.precomputed_fare_ore for t in addition.trips}
    assert fares == {model.floor_ore(o)}
    assert model.floor_ore(o) <= model.estimate_ore(o)


def test_no_pickup_is_generated_after_the_return_deadline(offers):
    o = next(o for o in offers if o.pickup.city == "Uppsala")
    service_date = o.available_at.date()
    addition = freerider_route_addition(
        o, service_date, FreeriderCostModel(), now=o.available_at
    )
    latest_pickup_seconds = to_service_seconds(o.latest_pickup(), service_date)
    assert all(t.departures[0] <= latest_pickup_seconds for t in addition.trips)

    return_deadline = to_service_seconds(o.latest_return, service_date)
    assert all(t.arrivals[1] <= return_deadline for t in addition.trips)


def test_expired_offer_yields_no_route(offers):
    o = offers[0]
    after_expiry = (o.expire_time or o.latest_return) + timedelta(minutes=1)
    assert freerider_route_addition(o, o.available_at.date(), FreeriderCostModel(), now=after_expiry) is None


def test_offer_not_yet_collectable_today_is_skipped(offers):
    """An offer whose window opens next week must not appear in today's timetable."""
    o = next(o for o in offers if o.pickup.city == "Uppsala")
    long_past = o.available_at.date() - timedelta(days=10)
    addition = freerider_route_addition(
        o, long_past, FreeriderCostModel(), now=o.available_at - timedelta(days=10)
    )
    assert addition is None


def test_now_truncates_the_window(offers):
    """A car cannot be collected in the past."""
    o = next(o for o in offers if o.pickup.city == "Uppsala")
    service_date = o.available_at.date()
    late = o.available_at + timedelta(hours=3)
    if late > o.latest_pickup():
        pytest.skip("fixture window too short for this case")
    addition = freerider_route_addition(o, service_date, FreeriderCostModel(), now=late)
    assert addition is not None
    first = addition.trips[0].departures[0]
    assert first >= to_service_seconds(late, service_date) - 1800


def test_trip_count_is_bounded(offers):
    """A multi-day window must not explode into thousands of synthetic trips."""
    longest = max(offers, key=lambda o: (o.latest_return - o.available_at))
    addition = freerider_route_addition(
        longest,
        longest.available_at.date(),
        FreeriderCostModel(),
        now=longest.available_at,
        step_minutes=5,
    )
    if addition is not None:
        assert len(addition.trips) <= 200


# --- integration with the router -----------------------------------------


def test_overlaid_freerider_offer_is_routable(offers):
    """End to end: a real offer becomes a synthetic route the router can ride."""
    base = Timetable(stops=[], routes=[], stop_routes=[], transfers=[], stop_index={})
    o = next(o for o in offers if o.pickup.city == "Uppsala")
    service_date = o.available_at.date()
    additions = freerider_additions(
        [o], service_date, FreeriderCostModel(), now=o.available_at
    )
    tt = overlay_timetable(base, additions)

    assert tt.num_stops == 2
    assert tt.validate() == []

    from tripps.routing.floors import PriceFloorModel

    depart = to_service_seconds(o.available_at, service_date)
    res = run_mcraptor(
        tt,
        PriceFloorModel(),
        RaptorQuery(
            origins=[(tt.index_of(o.pickup.stop_id), depart)],
            targets={tt.index_of(o.dropoff.stop_id)},
        ),
    )
    # A range query returns one label per collectable pickup time; the car is free at all
    # of them, and every one takes the same time to drive.
    assert len(res.labels) > 1
    assert {lbl.price_ore for lbl in res.labels} == {0}  # 33 km is free
    assert all(lbl.arrival - lbl.departure == o.drive_seconds for lbl in res.labels)
    assert min(lbl.departure for lbl in res.labels) >= depart


def test_overlay_preserves_base_routes(offers):
    """Injecting cars must not disturb the timetable built from GTFS."""
    from .support import Net, at, hhmm

    base = Net().route("R", ["STO", "GBG"], [[at(hhmm(8)), at(hhmm(12))]]).build()
    o = next(o for o in offers if o.pickup.city == "Uppsala")
    additions = freerider_additions(
        [o], o.available_at.date(), FreeriderCostModel(), now=o.available_at
    )
    tt = overlay_timetable(base, additions)

    assert len(tt.routes) == len(base.routes) + 1
    assert tt.num_stops == base.num_stops + 2
    assert tt.validate() == []
    # The original route still resolves and still has its trip.
    assert tt.routes[0].info.id == "R"
    assert len(tt.routes[0].trips) == 1
