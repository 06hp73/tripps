"""Freerider watcher: geographic matching, once-only hits, proactive route suggestions."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from tripps.db import Database
from tripps.ingest.freerider import parse_offers
from tripps.watcher import (
    Watch,
    hit_payload,
    match_watches,
    notify_webhook,
    upcoming_cars_on_route,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def offers():
    raw = json.loads((FIXTURES / "freerider_sweden.json").read_text(encoding="utf-8"))
    return parse_offers(raw)


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "w.sqlite3")
    yield database
    database.close()


def _watch_near(offer, *, radius_km=40.0, wid=1) -> Watch:
    """A watch centred on an offer's own pickup/dropoff, so it definitely matches."""
    return Watch(
        id=wid,
        origin=offer.pickup.city,
        destination=offer.dropoff.city,
        origin_lat=offer.pickup.lat,
        origin_lon=offer.pickup.lon,
        dest_lat=offer.dropoff.lat,
        dest_lon=offer.dropoff.lon,
        radius_km=radius_km,
    )


# --- matching --------------------------------------------------------------


def test_watch_matches_a_car_on_its_route(offers):
    offer = offers[0]
    assert _watch_near(offer).matches(offer)


def test_watch_ignores_a_car_going_elsewhere(offers):
    # A watch centred on one offer must not match an offer with a distant pickup.
    a, b = offers[0], next(o for o in offers if o.pickup.city != offers[0].pickup.city)
    watch = _watch_near(a, radius_km=20.0)
    assert not watch.matches(b) or b.pickup.city == a.pickup.city


def test_radius_bounds_the_match(offers):
    offer = offers[0]
    tight = Watch(
        id=1, origin="", destination="",
        origin_lat=offer.pickup.lat + 1.0, origin_lon=offer.pickup.lon,  # ~111 km north
        dest_lat=offer.dropoff.lat, dest_lon=offer.dropoff.lon, radius_km=40.0,
    )
    assert not tight.matches(offer)


def test_match_watches_pairs_every_matching_car(offers):
    watches = [_watch_near(offers[0], wid=1), _watch_near(offers[1], wid=2)]
    pairs = match_watches(watches, offers)
    assert any(w.id == 1 and o is offers[0] for w, o in pairs)
    assert any(w.id == 2 and o is offers[1] for w, o in pairs)


# --- once-only hits (the dedupe contract) ----------------------------------


def test_a_car_is_recorded_as_a_hit_only_once(db, offers):
    offer = offers[0]
    args = dict(
        route_id=offer.route_id, pickup=offer.pickup.name, dropoff=offer.dropoff.name,
        available_at=offer.available_at.isoformat(), car=offer.car_model,
    )
    assert db.record_hit(1, **args) is True, "first sighting announces"
    assert db.record_hit(1, **args) is False, "second sighting is silent"
    # A different watch for the same car announces independently.
    assert db.record_hit(2, **args) is True


def test_recent_hits_reads_back(db, offers):
    offer = offers[0]
    db.record_hit(
        1, route_id=offer.route_id, pickup=offer.pickup.name, dropoff=offer.dropoff.name,
        available_at=offer.available_at.isoformat(), car=offer.car_model,
    )
    hits = db.recent_hits()
    assert len(hits) == 1
    assert hits[0]["pickup"] == offer.pickup.name


# --- watch registry --------------------------------------------------------


def test_add_list_and_deactivate_watch(db):
    wid = db.add_watch(
        origin="Stockholm", destination="Göteborg",
        origin_lat=59.33, origin_lon=18.06, dest_lat=57.71, dest_lon=11.97,
        radius_km=40.0, webhook_url=None,
    )
    assert [w["id"] for w in db.list_watches()] == [wid]
    assert db.deactivate_watch(wid) is True
    assert db.list_watches(active_only=True) == []
    assert db.deactivate_watch(9999) is False


# --- proactive suggestions -------------------------------------------------


def test_upcoming_cars_on_route_finds_matches_sorted_by_date(offers):
    offer = offers[0]
    found = upcoming_cars_on_route(
        offers, offer.pickup.lat, offer.pickup.lon, offer.dropoff.lat, offer.dropoff.lon
    )
    assert offer in found
    dates = [o.available_at for o in found]
    assert dates == sorted(dates)


def test_upcoming_cars_excludes_unrelated_routes(offers):
    # A point in the ocean matches nothing.
    assert upcoming_cars_on_route(offers, 0.0, 0.0, 1.0, 1.0) == []


# --- webhook notification --------------------------------------------------


@respx.mock
async def test_webhook_posts_the_payload(offers):
    route = respx.post("https://example.test/hook").mock(return_value=httpx.Response(200))
    ok = await notify_webhook(
        "https://example.test/hook", hit_payload(_watch_near(offers[0]), offers[0])
    )
    assert ok is True
    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert "pickup" in body and "book_url" in body


@respx.mock
async def test_webhook_failure_is_swallowed(offers):
    respx.post("https://example.test/hook").mock(return_value=httpx.Response(500))
    ok = await notify_webhook("https://example.test/hook", hit_payload(_watch_near(offers[0]), offers[0]))
    assert ok is False  # reported, not raised


async def test_webhook_network_error_is_swallowed(offers):
    # No respx mock: the connection fails, and the function must still return False.
    ok = await notify_webhook(
        "http://127.0.0.1:9/nope", hit_payload(_watch_near(offers[0]), offers[0]), timeout=0.2
    )
    assert ok is False


def test_hit_payload_shape(offers):
    payload = hit_payload(_watch_near(offers[0]), offers[0])
    assert set(payload) >= {"watch_id", "route", "pickup", "dropoff", "available_at", "car", "book_url"}
    assert isinstance(payload["direct_km"], (int, float))
