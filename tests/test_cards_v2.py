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


# --- zone-combination hint (partial cross-border coverage) ------------------


def test_partially_covers_a_leg_that_crosses_the_border():
    cov = PassCoverage(_timetable(), cards={"sk": _card()})  # no border stops
    # Lund (in Skåne) -> Sölvesborg (Blekinge): exactly one endpoint in region, not covered.
    assert cov.partially_covers("sk", _leg("Öresundståg", LUND, SOLVESBORG))
    assert cov.partially_covers("sk", _leg("Öresundståg", SOLVESBORG, LUND))  # either direction


def test_fully_covered_or_fully_outside_is_not_partial():
    cov = PassCoverage(_timetable(), cards={"sk": _card()})
    assert not cov.partially_covers("sk", _leg("Öresundståg", MALMO, LUND))  # both in region
    assert not cov.partially_covers("sk", _leg("Öresundståg", SOLVESBORG, KARLSHAMN))  # both out


def test_a_seeded_border_stop_is_covered_not_merely_partial():
    cov = PassCoverage(_timetable(), cards={"sk": _card(border_stops=frozenset({"SOLV"}))})
    # With Sölvesborg seeded, the leg is fully covered, so it is not flagged as partial.
    assert cov.covers("sk", _leg("Öresundståg", LUND, SOLVESBORG))
    assert not cov.partially_covers("sk", _leg("Öresundståg", LUND, SOLVESBORG))


def test_partial_coverage_needs_an_honored_operator():
    cov = PassCoverage(_timetable(), cards={"sk": _card()})
    # Västtrafik is not honored by this card, so a border-crossing on it is not "partial".
    assert not cov.partially_covers("sk", _leg("Västtrafik", LUND, SOLVESBORG))


def test_seeded_border_stops_are_exactly_the_verified_set():
    # Cross-border is opt-in and verified per card. Only the four cards with an official,
    # cited FREE extension are seeded; every other card keeps an empty allow-list so none can
    # over-cover. This test pins the set so a careless edit that frees a paid leg fails loudly.
    from tripps.passes import load_cards

    seeded = {c.id: c.border_stops for c in load_cards().values() if c.border_stops}
    assert set(seeded) == {"ul-uppsala", "dalatrafik", "lanstrafiken-orebro", "vl-vastmanland", "x-trafik"}
    assert seeded["ul-uppsala"] == frozenset({"740000210", "740000027", "740000556"})
    assert seeded["x-trafik"] == frozenset({"740000704", "740000630", "740000111"})
    assert seeded["dalatrafik"] == frozenset(
        {"740020094", "740000218", "740000903", "740000280", "740000214",
         "740000195", "740000244", "740000638", "740001563"}
    )
    assert seeded["lanstrafiken-orebro"] == frozenset({"740000291", "740000186"})
    assert seeded["vl-vastmanland"] == frozenset({"740000133"})


def test_dalatrafik_ticket_valid_one_town_across_the_border_on_tib():
    # Falun (in Dalarna) -> Sala (Västmanland, a seeded border stop) on Tåg i Bergslagen.
    from tripps.passes import PassCoverage, load_cards

    FALUN = Stop(id="740000060", name="Falun C", lat=60.61, lon=15.63)
    SALA = Stop(id="740000214", name="Sala station", lat=59.92, lon=16.60)
    HOFORS = Stop(id="740000218", name="Hofors station", lat=60.55, lon=16.28)
    b = TimetableBuilder()
    for s in (FALUN, SALA, HOFORS):
        b.add_stop(s)
    b.add_trip(
        RouteInfo(id="tib", mode=TransportMode.TRAIN, operator="Tåg i Bergslagen"),
        ["740000060", "740000214"], Trip(id="t", arrivals=[0, 600], departures=[0, 600]),
    )
    agency_stops = {
        "Dalatrafik": ["740000060"],
        "Tåg i Bergslagen": ["740000060", "740000214", "740000218"],
    }
    cov = PassCoverage(b.build(), cards={"dalatrafik": load_cards()["dalatrafik"]}, agency_stops=agency_stops)
    assert cov.covers("dalatrafik", _leg("Tåg i Bergslagen", FALUN, SALA))  # in-region -> border stop
    assert cov.covers("dalatrafik", _leg("Tåg i Bergslagen", SALA, FALUN))  # reverse
    # Sala -> Hofors: neither endpoint in Dalarna (both border-side) -> not covered.
    assert not cov.covers("dalatrafik", _leg("Tåg i Bergslagen", SALA, HOFORS))


# UL cross-border seed: a real registry card exercised over a synthetic feed using the real
# GTFS stop_ids the registry references.
UPPSALA = Stop(id="740000005", name="Uppsala C", lat=59.86, lon=17.65)
GAVLE = Stop(id="740000210", name="Gävle Centralstation", lat=60.68, lon=17.13)
STOCKHOLM = Stop(id="740000001", name="Stockholm C", lat=59.33, lon=18.06)
ARLANDA = Stop(id="740000556", name="Arlanda Centralstation", lat=59.65, lon=17.93)


def _ul_coverage():
    from tripps.passes import PassCoverage, load_cards

    b = TimetableBuilder()
    for s in (UPPSALA, GAVLE, STOCKHOLM, ARLANDA):
        b.add_stop(s)
    b.add_trip(
        RouteInfo(id="ml", mode=TransportMode.TRAIN, operator="Mälartåg"),
        ["740000005", "740000556", "740000210"],
        Trip(id="m1", arrivals=[0, 600, 1200], departures=[0, 600, 1200]),
    )
    # UL's home agency serves only Uppsala here -> that is its region.
    agency_stops = {"UL": ["740000005"], "Mälartåg": ["740000005", "740000556", "740000210", "740000001"]}
    return PassCoverage(b.build(), cards={"ul-uppsala": load_cards()["ul-uppsala"]}, agency_stops=agency_stops)


def test_ul_ticket_valid_to_gavle_across_the_border():
    cov = _ul_coverage()
    assert cov.covers("ul-uppsala", _leg("Mälartåg", UPPSALA, GAVLE))
    assert cov.covers("ul-uppsala", _leg("Mälartåg", GAVLE, UPPSALA))  # reverse too


def test_ul_ticket_valid_to_arlanda_ride():
    cov = _ul_coverage()
    assert cov.covers("ul-uppsala", _leg("Mälartåg", UPPSALA, ARLANDA))


def test_ul_ticket_not_valid_onward_into_the_sl_network():
    cov = _ul_coverage()
    # Uppsala -> Stockholm: Stockholm is neither in region nor a border stop (needs a combo).
    assert not cov.covers("ul-uppsala", _leg("Mälartåg", UPPSALA, STOCKHOLM))
    # Gävle -> Stockholm: neither endpoint in region.
    assert not cov.covers("ul-uppsala", _leg("Mälartåg", GAVLE, STOCKHOLM))


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


# --- combined multi-card coverage (union across a through-leg's path) --------

SA = Stop(id="SA", name="Alfa", lat=55.0, lon=13.0)
SB = Stop(id="SB", name="Beta", lat=55.5, lon=13.2)
SC = Stop(id="SC", name="Gamma", lat=56.0, lon=13.4)
SD = Stop(id="SD", name="Delta", lat=56.5, lon=13.6)
SX = Stop(id="SX", name="Xeno", lat=57.0, lon=13.8)  # in no card's region


def _combo_cards():
    reg1 = TravelCard(
        id="reg1", name="Region One", region="R1",
        honored_operators=frozenset({"Kust"}), coverage_model="region-stops",
        region_agencies=frozenset({"Ag1"}),
    )
    reg2 = TravelCard(
        id="reg2", name="Region Two", region="R2",
        honored_operators=frozenset({"Kust"}), coverage_model="region-stops",
        region_agencies=frozenset({"Ag2"}),
    )
    return {"reg1": reg1, "reg2": reg2}


_COMBO_AGENCY_STOPS = {"Ag1": ["SA", "SB"], "Ag2": ["SC", "SD"], "Kust": ["SA", "SB", "SC", "SD"]}


def _combo_cov():
    b = TimetableBuilder()
    for s in (SA, SB, SC, SD, SX):
        b.add_stop(s)
    b.add_trip(
        RouteInfo(id="k", mode=TransportMode.TRAIN, operator="Kust"),
        ["SA", "SB", "SC", "SD"], Trip(id="k1", arrivals=[0, 6, 12, 18], departures=[0, 6, 12, 18]),
    )
    return PassCoverage(b.build(), cards=_combo_cards(), agency_stops=_COMBO_AGENCY_STOPS)


def _through(frm, to, via, operator="Kust"):
    dep = datetime(2026, 7, 22, 8, 0, tzinfo=TZ)
    return Leg(
        from_stop=frm, to_stop=to, mode=TransportMode.TRAIN, operator=operator,
        departure=dep, arrival=dep.replace(hour=10), service_ref="t", via_stop_ids=tuple(via),
    )


def test_two_cards_jointly_cover_a_through_leg():
    cov = _combo_cov()
    combined = cov.combined_cover(["reg1", "reg2"], _through(SA, SD, ["SA", "SB", "SC", "SD"]))
    assert combined is not None
    assert {c.id for c in combined} == {"reg1", "reg2"}


def test_combined_needs_the_whole_path_covered():
    cov = _combo_cov()
    # An intermediate stop (SX) in no held card's region -> not jointly covered.
    assert cov.combined_cover(["reg1", "reg2"], _through(SA, SD, ["SA", "SB", "SX", "SC", "SD"])) is None


def test_combined_needs_two_or_more_cards():
    cov = _combo_cov()
    # One card alone cannot reach SC/SD, so there is no combined cover.
    assert cov.combined_cover(["reg1"], _through(SA, SD, ["SA", "SB", "SC", "SD"])) is None


def test_combined_fails_closed_without_a_stop_path():
    cov = _combo_cov()
    # No via path -> the middle cannot be verified -> never freed (fail closed).
    assert cov.combined_cover(["reg1", "reg2"], _through(SA, SD, [])) is None


def test_combined_never_frees_a_globally_excluded_operator():
    cov = _combo_cov()
    leg = _through(SA, SD, ["SA", "SB", "SC", "SD"], operator="SJ")
    assert cov.combined_cover(["reg1", "reg2"], leg) is None


async def test_pass_adapter_prices_a_jointly_covered_leg_free():
    from tripps.passes import PassAdapter

    adapter = PassAdapter()
    adapter._coverage = _combo_cov()  # inject synthetic coverage
    adapter._held = frozenset({"reg1", "reg2"})
    leg = _through(SA, SD, ["SA", "SB", "SC", "SD"])
    assert adapter.supports(leg)
    quote = await adapter.quote_leg(leg)
    assert quote.amount_ore == 0
    assert "jointly cover" in (quote.note or "")
    assert "Region One" in quote.note and "Region Two" in quote.note


# --- SL operator-only over-coverage guard (denied_stops) --------------------

def test_sl_card_denies_pendeltag_to_uppsala_but_keeps_valid_legs():
    from tripps.passes import PassCoverage, load_cards

    STHLM = Stop(id="740000001", name="Stockholm C", lat=59.33, lon=18.06)
    UPPSALA = Stop(id="740000005", name="Uppsala C", lat=59.86, lon=17.64)
    KNIVSTA = Stop(id="740000559", name="Knivsta", lat=59.72, lon=17.79)
    SODERTALJE = Stop(id="740000615", name="Södertälje", lat=59.20, lon=17.63)
    b = TimetableBuilder()
    for s in (STHLM, UPPSALA, KNIVSTA, SODERTALJE):
        b.add_stop(s)
    b.add_trip(
        RouteInfo(id="pt40", mode=TransportMode.TRAIN, operator="SL"),
        ["740000001", "740000559", "740000005"], Trip(id="t", arrivals=[0, 300, 600], departures=[0, 300, 600]),
    )
    cov = PassCoverage(b.build())  # real registry, incl. sl-stockholm denied_stops

    def sl(frm, to):
        dep = datetime(2026, 7, 22, 8, 0, tzinfo=TZ)
        return Leg(from_stop=frm, to_stop=to, mode=TransportMode.TRAIN, operator="SL",
                   departure=dep, arrival=dep.replace(hour=9), service_ref="t")

    assert load_cards()["sl-stockholm"].denied_stops == frozenset({"740000559", "740000005"})
    assert not cov.covers("sl-stockholm", sl(STHLM, UPPSALA))   # to a denied stop
    assert not cov.covers("sl-stockholm", sl(UPPSALA, STHLM))   # from a denied stop
    assert not cov.covers("sl-stockholm", sl(STHLM, KNIVSTA))   # Knivsta is denied too
    assert cov.covers("sl-stockholm", sl(STHLM, SODERTALJE))    # within SL validity, still free


# --- variant-gated products (Movingo / Norrlandsresan) never auto-free -------

def test_variant_gated_cards_never_auto_free():
    from tripps.passes import PassCoverage, honored_operators, load_cards

    cards = load_cards()
    assert cards["movingo"].variant_gated and cards["norrlandsresan"].variant_gated
    # Build a timetable where Movingo's union region would otherwise cover a Mälartåg leg.
    A = Stop(id="740000001", name="Stockholm C", lat=59.33, lon=18.06)
    B = Stop(id="740000004", name="Eskilstuna", lat=59.37, lon=16.51)
    b = TimetableBuilder()
    b.add_stop(A)
    b.add_stop(B)
    b.add_trip(RouteInfo(id="m", mode=TransportMode.TRAIN, operator="Mälartåg"),
               ["740000001", "740000004"], Trip(id="t", arrivals=[0, 600], departures=[0, 600]))
    agency_stops = {a: ["740000001", "740000004"] for a in ("SL", "UL", "VL", "Sörmlandstrafiken",
                    "Länstrafiken Örebro", "Östgötatrafiken", "Mälartåg")}
    cov = PassCoverage(b.build(), cards=cards, agency_stops=agency_stops)
    leg = _leg("Mälartåg", A, B)
    assert not cov.covers("movingo", leg)           # never auto-freed despite union coverage
    assert not cov.is_supported("movingo")          # marked hint-only
    assert honored_operators(["movingo"]) == frozenset()  # not even floor-zeroed


def test_itinerary_co2_estimate():
    from tripps.models import Itinerary, PriceConfidence, Quote

    def priced_leg(op, mode, frm, to, amount):
        dep = datetime(2026, 7, 22, 8, 0, tzinfo=TZ)
        return Leg(from_stop=frm, to_stop=to, mode=mode, operator=op, departure=dep,
                   arrival=dep.replace(hour=11), service_ref="t",
                   quote=Quote(source="x", amount_ore=amount, confidence=PriceConfidence.EXACT))

    STO = Stop(id="STO", name="Stockholm", lat=59.33, lon=18.06)
    GBG = Stop(id="GBG", name="Göteborg", lat=57.71, lon=11.97)
    train = Itinerary(legs=[priced_leg("SJ", TransportMode.TRAIN, STO, GBG, 30000)])
    flight = Itinerary(legs=[priced_leg("SAS", TransportMode.FLIGHT, STO, GBG, 90000)])
    # ~400 km great-circle: train ~1 g/km is tiny, flight ~127 g/km is large.
    assert 0 < train.co2_grams < flight.co2_grams
    assert flight.co2_grams > 40_000  # ~400 km * 127 g ~= 50 kg
    assert "co2_grams" in train.model_dump()
