"""The validation harness's invariant classification (pass/warn/fail), with a fake planner."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from tripps.models import (
    Itinerary,
    Leg,
    PriceConfidence,
    Quote,
    SearchResponse,
    Stop,
    TransportMode,
)
from tripps.validation import CORRIDORS, Corridor, run_validation, summarize, validate_one

TZ = ZoneInfo("Europe/Stockholm")
A = Stop(id="A", name="Stockholm C", lat=59.33, lon=18.06)
B = Stop(id="B", name="Göteborg C", lat=57.71, lon=11.97)
DAY = date(2026, 7, 22)
CORR = Corridor("A->B", "Stockholm C", "Göteborg C", frozenset({TransportMode.TRAIN}))


def _itin(amount):
    dep = datetime(2026, 7, 22, 8, 0, tzinfo=TZ)
    quote = (
        Quote(source="tora", amount_ore=amount, confidence=PriceConfidence.EXACT)
        if amount is not None
        else Quote.unavailable("operator-link")
    )
    leg = Leg(from_stop=A, to_stop=B, mode=TransportMode.TRAIN, operator="SJ",
              departure=dep, arrival=dep.replace(hour=11), service_ref="t", quote=quote)
    return Itinerary(legs=[leg])


class _FakePlanner:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    async def search(self, *a, **k):
        if self._exc is not None:
            raise self._exc
        return self._response, object()


def _make(response=None, exc=None):
    return lambda _day: _FakePlanner(response, exc)


def _response(itins, warnings=None):
    return SearchResponse(
        origin=A, destination=B, date=DAY.isoformat(),
        itineraries=itins, warnings=warnings or [], source_status={"tora": "ok"},
    )


async def test_pass_when_priced_and_no_violation():
    r = await validate_one(_make(_response([_itin(45000)])), CORR, DAY, None)
    assert r.status == "pass"
    assert r.cheapest_sek == 450.0 and r.priced == 1


async def test_fail_when_no_itinerary():
    r = await validate_one(_make(_response([])), CORR, DAY, None)
    assert r.status == "fail"
    assert "no itinerary" in r.messages[0]


async def test_fail_on_price_floor_violation():
    resp = _response([_itin(45000)], warnings=["2 price-floor violation(s) detected; a cheaper itinerary may have been pruned"])
    r = await validate_one(_make(resp), CORR, DAY, None)
    assert r.status == "fail"
    assert any("floor" in m.lower() for m in r.messages)


async def test_warn_when_routed_but_unpriced():
    r = await validate_one(_make(_response([_itin(None)])), CORR, DAY, None)
    assert r.status == "warn"
    assert r.priced == 0 and r.itineraries == 1


async def test_fail_when_search_raises():
    r = await validate_one(_make(exc=LookupError("unknown place")), CORR, DAY, None)
    assert r.status == "fail"
    assert "search error" in r.messages[0]


async def test_run_validation_and_summarize():
    corridors = [CORR, CORR, CORR]
    responses = iter([_response([_itin(45000)]), _response([]), _response([_itin(None)])])

    def make(_day):
        return _FakePlanner(next(responses))

    results = await run_validation(make, service_date=DAY, offers=None, corridors=corridors)
    passed, warned, failed = summarize(results)
    assert (passed, warned, failed) == (1, 1, 1)


def test_corridor_registry_is_sane():
    assert len(CORRIDORS) >= 5
    assert all(c.origin and c.destination and c.modes for c in CORRIDORS)
