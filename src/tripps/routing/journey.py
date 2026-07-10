"""Turn a McRAPTOR label chain back into an `Itinerary` of concrete legs.

The router works in stop indices and service-day seconds because that is what makes the
rounds fast. Everything the user sees is wall-clock datetimes and named stops, so the
translation happens exactly once, here, at the boundary.
"""

from __future__ import annotations

from datetime import date

from ..models import Itinerary, Leg, TransportMode
from ..timeutil import from_service_seconds
from .mcraptor import Label, RideEdge, TransferEdge, unwind
from .timetable import Timetable


def label_to_itinerary(tt: Timetable, label: Label, service_date: date) -> Itinerary:
    """Rebuild the journey that produced `label`, origin first."""
    legs: list[Leg] = []
    chain = unwind(label)

    for previous, node in zip(chain, chain[1:], strict=False):
        edge = node.edge
        if isinstance(edge, RideEdge):
            legs.append(_ride_leg(tt, edge, service_date))
        elif isinstance(edge, TransferEdge):
            legs.append(_walk_leg(tt, edge, node, previous, service_date))
        else:  # pragma: no cover - a non-origin label always carries an edge
            raise ValueError(f"label at stop {node.stop} has no edge")

    if not legs:
        raise ValueError("cannot build an itinerary from an origin-only label")
    return Itinerary(legs=_tighten_leading_walks(legs), floor_price_ore=label.price_ore)


def _tighten_leading_walks(legs: list[Leg]) -> list[Leg]:
    """Delay the walk to the first vehicle so it ends exactly at boarding.

    RAPTOR relaxes footpaths the instant a stop is reached, so a journey that starts at the
    query time and boards a car six hours later is reconstructed as "walk at 00:00, then
    stand at the pickup point until 06:00". The traveller does not do that, and an itinerary
    that claims to depart at midnight and last 6h43m is simply wrong about both.

    Only the walks *before* the first vehicle move. A walk between two vehicles is a real
    connection whose slack is genuine waiting at a station, and a walk after the last vehicle
    already starts when the traveller alights.
    """
    first_vehicle = next((i for i, leg in enumerate(legs) if leg.is_transit), None)
    if first_vehicle in (None, 0):
        return legs

    tightened = list(legs)
    boarding = tightened[first_vehicle].departure
    for i in range(first_vehicle - 1, -1, -1):
        walk = tightened[i]
        duration = walk.arrival - walk.departure
        start = boarding - duration
        if start < walk.departure:
            break  # cannot start earlier than the query allows; leave the rest as they are
        tightened[i] = walk.model_copy(update={"departure": start, "arrival": boarding})
        boarding = start
    return tightened


def _ride_leg(tt: Timetable, edge: RideEdge, service_date: date) -> Leg:
    route = tt.routes[edge.route_idx]
    trip = route.trips[edge.trip_idx]
    return Leg(
        from_stop=tt.stops[route.stops[edge.board_pos]],
        to_stop=tt.stops[route.stops[edge.alight_pos]],
        mode=route.info.mode,
        operator=route.info.operator,
        departure=from_service_seconds(trip.departures[edge.board_pos], service_date),
        arrival=from_service_seconds(trip.arrivals[edge.alight_pos], service_date),
        service_ref=trip.id,
        headsign=trip.headsign,
    )


def _walk_leg(
    tt: Timetable, edge: TransferEdge, node: Label, previous: Label, service_date: date
) -> Leg:
    return Leg(
        from_stop=tt.stops[edge.from_stop],
        to_stop=tt.stops[node.stop],
        mode=TransportMode.WALK,
        operator=None,
        departure=from_service_seconds(previous.arrival, service_date),
        arrival=from_service_seconds(node.arrival, service_date),
        service_ref=None,
    )


def leg_distance_km(leg: Leg) -> float:
    from .timetable import haversine_km

    return haversine_km(
        leg.from_stop.lat, leg.from_stop.lon, leg.to_stop.lat, leg.to_stop.lon
    )


def dedupe_itineraries(itineraries: list[Itinerary]) -> list[Itinerary]:
    """Collapse itineraries that ride exactly the same vehicles at the same times.

    Two labels can describe the same journey when a footpath of zero benefit sits in the
    middle of one of them. The user does not care; showing the same trip twice looks broken.
    """
    seen: set[tuple] = set()
    unique: list[Itinerary] = []
    for itin in itineraries:
        key = tuple(
            (leg.mode.value, leg.from_stop.id, leg.to_stop.id, leg.service_ref, leg.departure)
            for leg in itin.legs
            if leg.mode is not TransportMode.WALK
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(itin)
    return unique


def pattern_key(itin: Itinerary) -> tuple:
    """Identity of the journey's *shape*: which services, between which stops, ignoring time.

    The range query returns one itinerary per departure, which is what lets a cheap late
    coach be discovered. But sixteen pickup times for the same free Hertz car are sixteen
    presentations of one journey, and pricing each of them separately would spend the call
    budget learning the same number over and over.
    """
    return tuple(
        (leg.mode.value, leg.operator or "", leg.from_stop.id, leg.to_stop.id)
        for leg in itin.legs
        if leg.mode is not TransportMode.WALK
    )


def spread_by_departure(
    itineraries: list[Itinerary], max_per_pattern: int
) -> list[Itinerary]:
    """Keep at most `max_per_pattern` departures of each journey shape, spread over the day.

    Taking the first N departures would price six morning buses and never reach the cheap
    night one, which defeats the point of the range query. Evenly spacing the picks across
    the sorted departures samples the price curve instead of its left edge.
    """
    if max_per_pattern <= 0:
        return itineraries
    groups: dict[tuple, list[Itinerary]] = {}
    for itin in itineraries:
        groups.setdefault(pattern_key(itin), []).append(itin)

    kept: list[Itinerary] = []
    for group in groups.values():
        group.sort(key=lambda i: i.departure)
        if len(group) <= max_per_pattern:
            kept.extend(group)
            continue
        step = (len(group) - 1) / (max_per_pattern - 1) if max_per_pattern > 1 else 1
        indices = sorted({int(round(i * step)) for i in range(max_per_pattern)})
        kept.extend(group[i] for i in indices)
    return kept


def collapse_equivalent(itineraries: list[Itinerary]) -> list[Itinerary]:
    """After pricing, fold journeys that are the same shape at the same price.

    Sixteen departures of one free Freerider car all cost zero; the traveller wants to see
    the car once, at the time it is soonest collectable. Two FlixBus departures at 800 and
    420 SEK are genuinely different offers and both survive.
    """
    best: dict[tuple, Itinerary] = {}
    order: list[tuple] = []
    for itin in itineraries:
        key = (pattern_key(itin), itin.total_price_ore)
        current = best.get(key)
        if current is None:
            best[key] = itin
            order.append(key)
        elif itin.arrival < current.arrival:
            best[key] = itin
    return [best[key] for key in order]
