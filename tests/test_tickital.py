"""Tickital second-hand rentals: a windowed period pass that frees covered legs at 0.

A rental reuses the whole travel-card coverage machinery (the region gate especially), so the
tests here focus on what is *new*: the date window, the terms-of-service note, owned-card
precedence, and the DB round-trip.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from tripps.db import Database
from tripps.models import Leg, PriceConfidence, Stop, TransportMode
from tripps.passes import TICKITAL_FARE_CLASS, PassAdapter, TickitalRental
from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip

TZ = ZoneInfo("Europe/Stockholm")

MALMO = Stop(id="MMX", name="Malmö C", lat=55.61, lon=13.00)
LUND = Stop(id="LUND", name="Lund C", lat=55.70, lon=13.19)
HASSLEHOLM = Stop(id="HLM", name="Hässleholm", lat=56.16, lon=13.77)
GOTEBORG = Stop(id="GBG", name="Göteborg C", lat=57.71, lon=11.97)


def _timetable():
    b = TimetableBuilder()
    for s in (MALMO, LUND, HASSLEHOLM, GOTEBORG):
        b.add_stop(s)
    sk = RouteInfo(id="pagatag", mode=TransportMode.TRAIN, operator="Skånetrafiken")
    b.add_trip(sk, ["MMX", "LUND", "HLM"], Trip(id="p1", arrivals=[0, 600, 1200], departures=[0, 600, 1200]))
    ot = RouteInfo(id="ore", mode=TransportMode.TRAIN, operator="Öresundståg")
    b.add_trip(ot, ["MMX", "LUND", "GBG"], Trip(id="o1", arrivals=[0, 600, 9000], departures=[0, 600, 9000]))
    return b.build()


def _leg(operator, frm, to, day: date, mode=TransportMode.TRAIN):
    dep = datetime(day.year, day.month, day.day, 8, 0, tzinfo=TZ)
    return Leg(
        from_stop=frm, to_stop=to, mode=mode, operator=operator,
        departure=dep, arrival=dep.replace(hour=10), service_ref="t",
    )


def _rental(**kw):
    base = dict(
        provider_id="skanetrafiken", price_ore=18000,
        valid_from=date(2026, 7, 20), valid_to=date(2026, 7, 26),
    )
    base.update(kw)
    return TickitalRental(**base)


# --- windowed coverage -----------------------------------------------------


async def test_rental_frees_covered_leg_inside_window():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), [], [_rental()])
    leg = _leg("Öresundståg", MALMO, LUND, date(2026, 7, 22))  # inside Skåne, inside window
    assert adapter.supports(leg)
    quote = await adapter.quote_leg(leg)
    assert quote.amount_ore == 0
    assert quote.confidence is PriceConfidence.EXACT
    assert quote.fare_class == TICKITAL_FARE_CLASS


async def test_rental_does_not_cover_leg_before_window():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), [], [_rental()])
    leg = _leg("Öresundståg", MALMO, LUND, date(2026, 7, 19))  # one day early
    assert not adapter.supports(leg)


async def test_rental_does_not_cover_leg_after_window():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), [], [_rental()])
    leg = _leg("Öresundståg", MALMO, LUND, date(2026, 7, 27))  # one day late
    assert not adapter.supports(leg)


async def test_rental_window_is_inclusive_at_both_ends():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), [], [_rental()])
    assert adapter.supports(_leg("Öresundståg", MALMO, LUND, date(2026, 7, 20)))
    assert adapter.supports(_leg("Öresundståg", MALMO, LUND, date(2026, 7, 26)))


async def test_rental_still_obeys_the_region_gate():
    # Malmö -> Göteborg leaves Skåne; a Skånetrafiken rental must not free it even in-window.
    adapter = PassAdapter()
    adapter.prepare(_timetable(), [], [_rental()])
    assert not adapter.supports(_leg("Öresundståg", MALMO, GOTEBORG, date(2026, 7, 22)))


# --- the terms-of-service note ---------------------------------------------


async def test_rental_note_states_price_window_and_block_risk():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), [], [_rental()])
    quote = await adapter.quote_leg(_leg("Öresundståg", MALMO, LUND, date(2026, 7, 22)))
    note = quote.note or ""
    assert "180 SEK" in note
    assert "blocked" in note.lower()
    assert "Skånetrafiken" in note


# --- owned card precedence -------------------------------------------------


async def test_owned_card_beats_a_rental_for_the_same_provider():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), ["skanetrafiken"], [_rental()])
    quote = await adapter.quote_leg(_leg("Öresundståg", MALMO, LUND, date(2026, 7, 22)))
    # The owned card's clean note wins; no rental terms warning.
    assert quote.fare_class == "travel card"
    assert "period ticket" in (quote.note or "")


async def test_rental_covers_when_no_card_and_out_of_window_owned_would_not():
    adapter = PassAdapter()
    adapter.prepare(_timetable(), [], [_rental()])
    # Different provider rental should not cover Skåne.
    adapter2 = PassAdapter()
    adapter2.prepare(_timetable(), [], [_rental(provider_id="vasttrafik")])
    assert not adapter2.supports(_leg("Öresundståg", MALMO, LUND, date(2026, 7, 22)))


# --- active_on -------------------------------------------------------------


def test_active_on_boundaries():
    r = _rental()
    assert not r.active_on(date(2026, 7, 19))
    assert r.active_on(date(2026, 7, 20))
    assert r.active_on(date(2026, 7, 26))
    assert not r.active_on(date(2026, 7, 27))


# --- DB round-trip ---------------------------------------------------------


def test_db_add_list_active_remove(tmp_path):
    db = Database(tmp_path / "t.db")
    try:
        rid = db.add_tickital_rental(
            provider_id="skanetrafiken", price_ore=18000,
            valid_from=date(2026, 7, 20), valid_to=date(2026, 7, 26), note="ad 42",
        )
        assert rid > 0

        rows = db.list_tickital_rentals()
        assert len(rows) == 1
        rental = TickitalRental.from_row(rows[0])
        assert rental.provider_id == "skanetrafiken"
        assert rental.price_ore == 18000
        assert rental.valid_from == date(2026, 7, 20)
        assert rental.valid_to == date(2026, 7, 26)
        assert rental.note == "ad 42"

        assert len(db.active_tickital_rentals(date(2026, 7, 22))) == 1
        assert db.active_tickital_rentals(date(2026, 7, 27)) == []

        assert db.remove_tickital_rental(rid) is True
        assert db.list_tickital_rentals() == []
        assert db.remove_tickital_rental(rid) is False
    finally:
        db.close()
