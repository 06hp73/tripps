"""v2.1 card refinements: partial-zone holdings and the cross-border boundary allow-list."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from tripps.db import Database
from tripps.models import Leg, Stop, TransportMode
from tripps.passes import PassCoverage, TravelCard
from tripps.routing.timetable import RouteInfo, TimetableBuilder, Trip

TZ = ZoneInfo("Europe/Stockholm")

MALMO = Stop(id="MMX", name="Malmö C", lat=55.61, lon=13.00)
LUND = Stop(id="LUND", name="Lund C", lat=55.70, lon=13.19)
SOLVESBORG = Stop(id="SOLV", name="Sölvesborg", lat=56.05, lon=14.58)  # first stop in Blekinge
KARLSHAMN = Stop(id="KARL", name="Karlshamn", lat=56.17, lon=14.86)  # deeper in Blekinge


def _timetable():
    b = TimetableBuilder()
    for s in (MALMO, LUND, SOLVESBORG, KARLSHAMN):
        b.add_stop(s)
    # Skånetrafiken (home agency) serves only the Skåne stops -> that is the region set.
    sk = RouteInfo(id="pag", mode=TransportMode.TRAIN, operator="Skånetrafiken")
    b.add_trip(sk, ["MMX", "LUND"], Trip(id="p1", arrivals=[0, 600], departures=[0, 600]))
    # Öresundståg (consortium) runs on into Blekinge.
    ot = RouteInfo(id="ore", mode=TransportMode.TRAIN, operator="Öresundståg")
    b.add_trip(ot, ["MMX", "LUND", "SOLV", "KARL"], Trip(id="o1", arrivals=[0, 6, 12, 18], departures=[0, 6, 12, 18]))
    return b.build()


def _leg(operator, frm, to):
    dep = datetime(2026, 7, 22, 8, 0, tzinfo=TZ)
    return Leg(
        from_stop=frm, to_stop=to, mode=TransportMode.TRAIN, operator=operator,
        departure=dep, arrival=dep.replace(hour=9), service_ref="t",
    )


def _card(border_stops=frozenset()):
    return TravelCard(
        id="sk", name="Skånetrafiken", region="Skåne",
        honored_operators=frozenset({"Skånetrafiken", "Öresundståg"}),
        coverage_model="region-stops", region_agencies=frozenset({"Skånetrafiken"}),
        border_stops=border_stops,
    )


# --- cross-border boundary allow-list --------------------------------------


def test_no_border_stops_means_leaving_the_region_is_not_covered():
    cov = PassCoverage(_timetable(), cards={"sk": _card()})
    assert cov.covers("sk", _leg("Öresundståg", MALMO, LUND))  # both in Skåne
    assert not cov.covers("sk", _leg("Öresundståg", LUND, SOLVESBORG))  # into Blekinge


def test_border_stop_extends_coverage_one_named_stop_over():
    cov = PassCoverage(_timetable(), cards={"sk": _card(border_stops=frozenset({"SOLV"}))})
    # Lund (in region) -> Sölvesborg (the named boundary stop) is now covered...
    assert cov.covers("sk", _leg("Öresundståg", LUND, SOLVESBORG))
    # ...and the reverse direction too.
    assert cov.covers("sk", _leg("Öresundståg", SOLVESBORG, LUND))


def test_border_stop_does_not_free_a_leg_deeper_into_the_neighbour():
    cov = PassCoverage(_timetable(), cards={"sk": _card(border_stops=frozenset({"SOLV"}))})
    # Sölvesborg (border) -> Karlshamn (deeper in Blekinge): neither endpoint in region.
    assert not cov.covers("sk", _leg("Öresundståg", SOLVESBORG, KARLSHAMN))
    # Lund -> Karlshamn: Karlshamn is not the named boundary stop.
    assert not cov.covers("sk", _leg("Öresundståg", LUND, KARLSHAMN))


def test_registry_cards_have_empty_border_stops_by_default():
    from tripps.passes import load_cards

    assert all(not c.border_stops for c in load_cards().values())


# --- partial-zone holdings -------------------------------------------------


def test_partial_card_excluded_from_coverage_but_still_listed(tmp_path):
    db = Database(tmp_path / "t.db")
    try:
        db.add_card("skanetrafiken")  # all-zone (default)
        db.add_card("vasttrafik", all_zone=False)  # partial
        assert db.list_cards() == ["skanetrafiken"]  # only all-zone frees legs
        rows = {r["provider_id"]: r["all_zone"] for r in db.list_card_rows()}
        assert rows == {"skanetrafiken": 1, "vasttrafik": 0}
    finally:
        db.close()


def test_re_adding_a_card_updates_its_zone_flag(tmp_path):
    db = Database(tmp_path / "t.db")
    try:
        db.add_card("skanetrafiken", all_zone=False)
        assert db.list_cards() == []  # partial: not covering
        db.add_card("skanetrafiken", all_zone=True)  # upgrade to all-zone
        assert db.list_cards() == ["skanetrafiken"]
    finally:
        db.close()


def test_all_zone_column_migrated_onto_an_old_db(tmp_path):
    # A DB created before the all_zone column existed.
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE travel_card (provider_id TEXT PRIMARY KEY, added_at TEXT NOT NULL)")
    conn.execute("INSERT INTO travel_card VALUES ('skanetrafiken', '2026-01-01T00:00:00')")
    conn.commit()
    conn.close()

    db = Database(path)  # __init__ runs the migration
    try:
        assert db.list_cards() == ["skanetrafiken"]  # existing rows default to all-zone
        assert db.list_card_rows()[0]["all_zone"] == 1
    finally:
        db.close()
