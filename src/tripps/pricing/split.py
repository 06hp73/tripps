"""Split-ticketing: is buying two SJ tickets, broken at a major hub, cheaper than the through fare?

SJ prices the *cheapest advance fare* per origin/destination independently, and the number of
cheap seats in each price bucket is set per segment. So on a busy through train the cheapest
Stockholm->Malmo fare can exceed the sum of the cheapest Stockholm->Alvesta and cheapest
Alvesta->Malmo fares - buy the two and stay in your seat. Measured savings on that corridor run
15-65 SEK a departure.

This module only proposes *where* a split could pay off (a curated set of major junctions the
train actually calls at). The orchestrator prices the two halves on the same physical train and
keeps the result as an advisory - `Quote.split_hint` - never as the itinerary's authoritative
price: two tickets are two contracts with no rebooking or delay protection across the break, so
the number we stand behind stays the bookable through fare. Honest tip, not a quiet re-price.
"""

from __future__ import annotations

from ..models import Leg, TransportMode
from ..models import Stop as _Stop

# The junctions worth testing a split at: major SJ long-distance calling points where separate
# regional/inter-city fare buckets make a break pay off. GTFS ids + real coordinates (SJ prices
# by GTFS id and ignores the coordinates; they are here so a sub-leg Stop is a faithful station,
# not a fiction). Keeping this curated bounds the extra price calls to known-useful hubs rather
# than every intermediate stop.
_HUBS: tuple[tuple[str, str, float, float], ...] = (
    ("740000004", "Alvesta station", 56.898782, 14.556318),
    ("740000006", "Hässleholm Centralstation", 56.157760, 13.763138),
    ("740000007", "Norrköping Centralstation", 58.596627, 16.183346),
    ("740000008", "Skövde Centralstation", 58.390897, 13.853193),
    ("740000009", "Linköping Centralstation", 58.416636, 15.624963),
    ("740000040", "Herrljunga station", 58.079188, 13.021275),
    ("740000050", "Nyköping Centralstation", 58.755686, 16.994783),
    ("740000077", "Hallsberg station", 59.066699, 15.110387),
    ("740000120", "Lund Centralstation", 55.708099, 13.186900),
    ("740000140", "Nässjö Centralstation", 57.652440, 14.693982),
    ("740000166", "Katrineholm Centralstation", 58.996593, 16.208325),
    ("740000180", "Mjölby station", 58.322982, 15.131987),
)

SPLIT_STATIONS: dict[str, _Stop] = {
    sid: _Stop(id=sid, name=name, lat=lat, lon=lon, modes=frozenset({TransportMode.TRAIN}))
    for sid, name, lat, lon in _HUBS
}

#: Most junctions to test on a single leg. Two keeps the worst-case extra calls small (each
#: split point costs up to two price lookups) while still covering both a northern and a
#: southern break on a long trip.
MAX_SPLITS_PER_LEG = 2


def _interior_hub_indices(leg: Leg) -> list[int]:
    """Indices in `leg.via_stop_ids` that are curated hubs strictly between the endpoints."""
    n = len(leg.via_stop_ids)
    if n < 3:
        return []
    return [i for i in range(1, n - 1) if leg.via_stop_ids[i] in SPLIT_STATIONS]


def _pick_indices(indices: list[int], n_stops: int) -> list[int]:
    """Choose at most MAX_SPLITS_PER_LEG hubs, preferring those nearest the leg's midpoint.

    A break near the middle balances the two halves, where an uneven fare split is most likely
    to undercut the through fare; the far ends rarely help.
    """
    mid = (n_stops - 1) / 2
    return sorted(sorted(indices, key=lambda i: abs(i - mid))[:MAX_SPLITS_PER_LEG])


def sub_legs(leg: Leg) -> list[tuple[str, Leg, Leg]]:
    """For each candidate hub on this train, the (hub_name, board->hub, hub->alight) sub-legs.

    Both sub-legs describe the SAME physical train (its own departure at the hub, arrival at
    the destination), so pricing them yields fares for a passenger who never changes seat.
    Returns an empty list when the leg carries no per-stop timing or no interior hub.
    """
    if not leg.via_stop_ids or len(leg.via_departures) != len(leg.via_stop_ids):
        return []
    if len(leg.via_arrivals) != len(leg.via_stop_ids):
        return []
    interior = _interior_hub_indices(leg)
    if not interior:
        return []

    out: list[tuple[str, Leg, Leg]] = []
    for i in _pick_indices(interior, len(leg.via_stop_ids)):
        hub = SPLIT_STATIONS[leg.via_stop_ids[i]]
        first = leg.model_copy(
            update={
                "to_stop": hub,
                "arrival": leg.via_arrivals[i],
                "quote": None,
                "via_stop_ids": leg.via_stop_ids[: i + 1],
                "via_departures": leg.via_departures[: i + 1],
                "via_arrivals": leg.via_arrivals[: i + 1],
                "path_km": None,  # the parent's ridden distance is wrong for a half
            }
        )
        second = leg.model_copy(
            update={
                "from_stop": hub,
                "departure": leg.via_departures[i],
                "quote": None,
                "via_stop_ids": leg.via_stop_ids[i:],
                "via_departures": leg.via_departures[i:],
                "via_arrivals": leg.via_arrivals[i:],
                "path_km": None,  # the parent's ridden distance is wrong for a half
            }
        )
        out.append((hub.name, first, second))
    return out
