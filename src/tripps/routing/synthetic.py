"""Turn one-off, window-based offers into scheduled routes the router already understands.

A Hertz Freerider car is not a scheduled service: it is "collect this car at Borlänge any
time between Wed 04:45 and Fri 17:45, and have it in Norrköping by the deadline". RAPTOR
has no concept of that. Rather than teach the algorithm about continuous availability
windows, the window is discretized into synthetic trips at a fixed step, which makes a
Freerider offer structurally identical to a bus that departs every 30 minutes.

This is why the router needs no Freerider-specific code path at all. Two properties make
the reduction sound:

* Every synthetic trip on one offer has the same duration, so no trip overtakes another
  and RAPTOR's route-scan assumption holds by construction.
* The fare is attached to the trip (`precomputed_fare_ore`), so the price criterion picks
  it up without a floor model.

Freerider offers are deliberately NOT modelled as footpaths. RAPTOR relaxes footpaths once
per round and assumes they are always available and transitively closed; a car that exists
only inside a two-day window satisfies neither, and encoding it as a footpath would let the
router "walk" a 300 km leg at any hour of the day.

Discretization is a lower bound on quality, never a source of infeasible answers: a coarse
step can miss the ideal pickup minute, but every trip it emits is genuinely collectable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from ..ingest.freerider import (
    FREERIDER_OPERATOR,
    FreeriderCostModel,
    FreeriderOffer,
)
from ..models import TransportMode
from ..timeutil import to_service_seconds
from .timetable import RouteAddition, RouteInfo, Trip

#: How finely the pickup window is sampled. 30 minutes keeps the trip count modest
#: (a 3-day window yields ~144 trips) while landing close to any desired departure.
DEFAULT_STEP_MINUTES = 30

#: Never generate pickups further ahead than this from the start of the service day.
#: Offers routinely span 2-3 days; a query for one day should not carry all of them.
DEFAULT_HORIZON_HOURS = 36

#: A Freerider offer is only worth routing if it is bookable now.
MAX_TRIPS_PER_OFFER = 200


def freerider_route_addition(
    offer: FreeriderOffer,
    service_date: date,
    cost_model: FreeriderCostModel,
    *,
    now: datetime,
    step_minutes: int = DEFAULT_STEP_MINUTES,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> RouteAddition | None:
    """Discretize one offer into a two-stop synthetic route, or None if unusable today.

    The fare on each trip is the cost model's *floor*, not its estimate: this value feeds
    McRAPTOR's price criterion, where the contract is that it must never exceed the true
    cost. Phase-2 pricing replaces it with the (higher, human-facing) estimate.
    """
    if not offer.is_live(now):
        return None
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")

    drive = timedelta(seconds=offer.drive_seconds)
    earliest = max(offer.available_at, now)
    latest = offer.latest_pickup()
    if offer.expire_time is not None:
        # The car cannot be collected after the offer stops being bookable.
        latest = min(latest, offer.expire_time)
    if latest < earliest:
        return None

    day_start = datetime.combine(service_date, datetime.min.time()).replace(
        tzinfo=earliest.tzinfo
    )
    window_start = max(earliest, day_start)
    window_end = min(latest, day_start + timedelta(hours=horizon_hours))
    if window_end < window_start:
        return None

    step = timedelta(minutes=step_minutes)
    fare = cost_model.floor_ore(offer)

    trips: list[Trip] = []
    pickup = _align_to_step(window_start, step_minutes)
    if pickup < window_start:
        pickup += step
    while pickup <= window_end and len(trips) < MAX_TRIPS_PER_OFFER:
        depart = to_service_seconds(pickup, service_date)
        arrive = to_service_seconds(pickup + drive, service_date)
        trips.append(
            Trip(
                id=f"freerider:{offer.route_id}@{depart}",
                arrivals=[depart, arrive],
                departures=[depart, arrive],
                headsign=offer.dropoff.name,
                precomputed_fare_ore=fare,
            )
        )
        pickup += step

    # A window shorter than one step still yields exactly one collectable car.
    if not trips:
        depart = to_service_seconds(window_start, service_date)
        arrive = to_service_seconds(window_start + drive, service_date)
        trips.append(
            Trip(
                id=f"freerider:{offer.route_id}@{depart}",
                arrivals=[depart, arrive],
                departures=[depart, arrive],
                headsign=offer.dropoff.name,
                precomputed_fare_ore=fare,
            )
        )

    return RouteAddition(
        info=RouteInfo(
            id=f"freerider:{offer.route_id}",
            mode=TransportMode.FREERIDER,
            operator=FREERIDER_OPERATOR,
            synthetic=True,
        ),
        stops=[offer.pickup.to_stop(), offer.dropoff.to_stop()],
        trips=trips,
    )


def _align_to_step(moment: datetime, step_minutes: int) -> datetime:
    """Round down to the previous step boundary, so pickups land on tidy clock times."""
    minute = (moment.minute // step_minutes) * step_minutes
    return moment.replace(minute=minute, second=0, microsecond=0)


def freerider_additions(
    offers: list[FreeriderOffer],
    service_date: date,
    cost_model: FreeriderCostModel,
    *,
    now: datetime,
    step_minutes: int = DEFAULT_STEP_MINUTES,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> list[RouteAddition]:
    additions = []
    for offer in offers:
        addition = freerider_route_addition(
            offer,
            service_date,
            cost_model,
            now=now,
            step_minutes=step_minutes,
            horizon_hours=horizon_hours,
        )
        if addition is not None:
            additions.append(addition)
    return additions


def connect_freerider_stations(
    offers: list[FreeriderOffer],
    stop_lookup,
    *,
    max_walk_km: float = 2.0,
    walk_speed_kmh: float = 4.5,
) -> list[tuple[str, str, int]]:
    """Footpaths linking Freerider stations to nearby timetabled stops.

    Without these a Freerider leg is an island: the router can reach the car only if the
    pickup station happens to coincide with a GTFS stop, which it almost never does. The
    walk is capped short because a Hertz depot 10 km out of town is not reachable on foot
    and pretending otherwise would invent infeasible itineraries.

    `stop_lookup(lat, lon, radius_km) -> list[tuple[Stop, float]]` supplies candidates.
    """
    from .timetable import haversine_km  # local import: avoids a cycle at module load

    edges: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    stations = {}
    for offer in offers:
        stations[offer.pickup.trac_code] = offer.pickup
        stations[offer.dropoff.trac_code] = offer.dropoff

    for station in stations.values():
        for stop, _distance in stop_lookup(station.lat, station.lon, max_walk_km):
            km = haversine_km(station.lat, station.lon, stop.lat, stop.lon)
            if km > max_walk_km:
                continue
            seconds = int(km / walk_speed_kmh * 3600)
            key = (station.stop_id, stop.id)
            if key in seen:
                continue
            seen.add(key)
            edges.append((station.stop_id, stop.id, max(seconds, 60)))
    return edges
