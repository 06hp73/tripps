"""Our car count must be explainable against hertzfreerider.se's own number.

The endpoint's `?country=SWEDEN` selects by *destination*, so its Sweden list includes cars
that start in Norway and drive in. `only_within` drops those, correctly — no Swedish
timetable reaches Sandnessjøen, so an itinerary starting there could never be built. But that
left us reporting 86 where the operator's page said 99, with nothing to explain the gap.
`crossing_into` counts exactly the difference so it can be stated instead of hidden.
"""

from __future__ import annotations

from tripps.ingest.freerider import crossing_into, only_within, parse_offers


def _station(code: str, name: str, country: str, lat: float, lon: float) -> dict:
    return {
        "tracCode": code, "name": name, "city": name, "country": country,
        "geoLat": lat, "geoLon": lon,
    }


def _route(rid: int, pickup: dict, dropoff: dict) -> dict:
    return {
        "id": rid, "transportOfferId": 1000 + rid,
        "pickupLocation": pickup, "returnLocation": dropoff,
        "availableAt": "2026-08-10T09:00:00", "latestReturn": "2026-08-12T21:00:00",
        "carModel": "VOLVO XC60", "distance": 600.0, "originalDistance": 500.0,
        "travelTime": 400, "originalTravelTime": 330,
    }


STO = _station("SWSTO60", "Stockholm", "SE", 59.33, 18.06)
GOT = _station("SWGOT60", "Göteborg", "SE", 57.71, 11.97)
MMA = _station("SWMMA60", "Malmö", "SE", 55.61, 13.00)
LLA = _station("SWLLA60", "Luleå", "SE", 65.58, 22.15)
SSJ = _station("NOSSJ60", "Sandnessjøen", "NO", 66.02, 12.63)
OSL = _station("NOOSL60", "Oslo", "NO", 59.91, 10.75)

# The endpoint groups by (pickup, dropoff); each route carries its own station objects.
RAW = [
    {"routes": [_route(1, STO, GOT), _route(2, GOT, STO)]},   # within Sweden
    {"routes": [_route(3, SSJ, LLA)]},                        # arrives from Norway
    {"routes": [_route(4, MMA, OSL)]},                        # leaves Sweden
]


def _offers():
    offers = parse_offers(RAW)
    assert len(offers) == 4, "fixture must parse cleanly, or the counts below prove nothing"
    return offers


def test_only_within_keeps_the_routable_ones() -> None:
    """Both endpoints in Sweden — the only shape the planner can actually use."""
    kept = only_within(_offers(), "se")
    assert len(kept) == 2
    assert all(o.pickup.country == "se" and o.dropoff.country == "se" for o in kept)


def test_crossing_into_counts_arrivals_only() -> None:
    """Ends in Sweden, starts abroad: what Hertz counts and we cannot route."""
    arriving = crossing_into(_offers(), "se")
    assert len(arriving) == 1
    assert arriving[0].pickup.country == "no"
    assert arriving[0].dropoff.country == "se"


def test_a_car_leaving_sweden_is_in_neither_bucket() -> None:
    """Malmö -> Oslo is not routable and is not an arrival; it must not pad either number."""
    offers = _offers()
    kept = only_within(offers, "se")
    arriving = crossing_into(offers, "se")
    leaving = [o for o in offers if o.pickup.country == "se" and o.dropoff.country != "se"]

    assert len(leaving) == 1
    assert not any(o in kept for o in leaving)
    assert not any(o in arriving for o in leaving)


def test_the_two_buckets_never_overlap() -> None:
    """They are reported as one sum, so double-counting would overstate availability."""
    offers = _offers()
    kept = {o.route_id for o in only_within(offers, "se")}
    arriving = {o.route_id for o in crossing_into(offers, "se")}
    assert kept.isdisjoint(arriving)
