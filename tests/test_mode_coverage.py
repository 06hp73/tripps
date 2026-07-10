"""Every selected mode is represented in results, even when one mode dominates on price."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tripps.models import (
    Itinerary,
    Leg,
    PriceConfidence,
    Quote,
    SearchConstraints,
    Stop,
    TransportMode,
)
from tripps.pricing.orchestrator import _cover_modes, _ensure_mode_coverage

TZ = ZoneInfo("Europe/Stockholm")


def _itin(mode, price_ore, hour):
    dep = datetime(2026, 7, 22, hour, 0, tzinfo=TZ)
    arr = datetime(2026, 7, 22, hour + 3, 0, tzinfo=TZ)
    leg = Leg(
        from_stop=Stop(id="A", name="A", lat=59.3, lon=18.0),
        to_stop=Stop(id="B", name="B", lat=57.7, lon=12.0),
        mode=mode,
        operator="op",
        departure=dep,
        arrival=arr,
        quote=Quote(source="s", amount_ore=price_ore, confidence=PriceConfidence.EXACT)
        if price_ore is not None else None,
    )
    return Itinerary(legs=[leg])


def test_cover_modes_reads_the_selected_travel_modes():
    c = SearchConstraints(allowed_modes=frozenset(
        {TransportMode.TRAIN, TransportMode.BUS, TransportMode.WALK, TransportMode.LOCAL_TRANSIT}
    ))
    assert _cover_modes(c) == [TransportMode.TRAIN, TransportMode.BUS]


def test_a_crowded_out_mode_is_pulled_in():
    # Four cheap buses then a pricier train; top-3 by price is all buses.
    ranked = [
        _itin(TransportMode.BUS, 30_000, 8),
        _itin(TransportMode.BUS, 32_000, 10),
        _itin(TransportMode.BUS, 34_000, 12),
        _itin(TransportMode.BUS, 36_000, 14),
        _itin(TransportMode.TRAIN, 50_000, 9),
    ]
    ranked.sort(key=lambda i: i.total_price_ore)
    shown = _ensure_mode_coverage(
        ranked, [TransportMode.TRAIN, TransportMode.BUS], max_results=3
    )
    modes = {leg.mode for i in shown for leg in i.legs}
    assert TransportMode.TRAIN in modes, "the train must be included though it fell outside top-3"
    assert TransportMode.BUS in modes
    # It lands in its correct price position (last, since it is priciest).
    assert shown[-1].total_price_ore == 50_000


def test_a_mode_with_no_priced_option_adds_nothing():
    ranked = [_itin(TransportMode.BUS, 30_000, 8), _itin(TransportMode.BUS, 32_000, 10)]
    shown = _ensure_mode_coverage(
        ranked, [TransportMode.TRAIN, TransportMode.BUS, TransportMode.FERRY], max_results=5
    )
    assert len(shown) == 2  # no train, no ferry existed, so nothing is invented


def test_unpriced_itineraries_are_not_used_as_representatives():
    ranked = [
        _itin(TransportMode.BUS, 30_000, 8),
        _itin(TransportMode.TRAIN, None, 9),  # unpriced train
    ]
    ranked.sort(key=lambda i: (i.total_price_ore is None, i.total_price_ore or 0))
    shown = _ensure_mode_coverage(ranked, [TransportMode.TRAIN, TransportMode.BUS], max_results=1)
    # The lone priced result is the bus; the unpriced train is not pulled in as coverage.
    assert all(i.total_price_ore is not None for i in shown if TransportMode.TRAIN in {leg.mode for leg in i.legs})


def test_top_n_is_preserved_when_all_modes_already_covered():
    ranked = [
        _itin(TransportMode.TRAIN, 30_000, 8),
        _itin(TransportMode.BUS, 32_000, 10),
        _itin(TransportMode.BUS, 34_000, 12),
    ]
    shown = _ensure_mode_coverage(
        ranked, [TransportMode.TRAIN, TransportMode.BUS], max_results=3
    )
    assert len(shown) == 3
    assert [i.total_price_ore for i in shown] == [30_000, 32_000, 34_000]
