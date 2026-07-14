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


# --- Skånetrafiken cross-county over-coverage guard (denied_stops) -----------

# Real GTFS ids, because the registry's denied_stops are keyed on them.
SK_MALMO = Stop(id="740000002", name="Malmö C", lat=55.609, lon=13.000)
SK_LUND = Stop(id="740000120", name="Lund C", lat=55.708, lon=13.187)
SK_OSBY = Stop(id="740000295", name="Osby", lat=56.380, lon=13.994)  # northern SKÅNE
SK_BASTAD = Stop(id="740001603", name="Båstad", lat=56.432, lon=12.907)  # SKÅNE
SK_SOLVESBORG = Stop(id="740000079", name="Sölvesborg", lat=56.050, lon=14.583)  # Blekinge
SK_ALMHULT = Stop(id="740000045", name="Älmhult", lat=56.551, lon=14.137)  # Kronoberg
SK_HALMSTAD = Stop(id="740000080", name="Halmstad C", lat=56.669, lon=12.865)  # Halland


def _skane_cov():
    from tripps.passes import load_cards

    stops = (SK_MALMO, SK_LUND, SK_OSBY, SK_BASTAD, SK_SOLVESBORG, SK_ALMHULT, SK_HALMSTAD)
    b = TimetableBuilder()
    for s in stops:
        b.add_stop(s)
    b.add_trip(
        RouteInfo(id="ore", mode=TransportMode.TRAIN, operator="Öresundståg"),
        [s.id for s in stops],
        Trip(id="o1", arrivals=list(range(0, 42, 6)), departures=list(range(0, 42, 6))),
    )
    # The agency's OWN vehicle reach includes the out-of-county stations (Pågatåg Nordost +
    # västkusten run there) - exactly the over-coverage the denied_stops must fence off.
    return PassCoverage(
        b.build(),
        cards={"skanetrafiken": load_cards()["skanetrafiken"]},
        agency_stops={"Skånetrafiken": [s.id for s in stops]},
    )


def test_skane_card_never_frees_out_of_county_stations():
    """The card's own trains reach Blekinge/Kronoberg/Halland, but the Skåne card is not
    valid there - freeing those legs would be lying about a real fare."""
    cov = _skane_cov()
    assert not cov.covers("skanetrafiken", _leg("Öresundståg", SK_LUND, SK_SOLVESBORG))
    assert not cov.covers("skanetrafiken", _leg("Öresundståg", SK_MALMO, SK_ALMHULT))
    assert not cov.covers("skanetrafiken", _leg("Öresundståg", SK_LUND, SK_HALMSTAD))
    assert not cov.covers("skanetrafiken", _leg("Öresundståg", SK_HALMSTAD, SK_MALMO))


def test_skane_card_still_covers_all_of_skane_including_the_north():
    cov = _skane_cov()
    assert cov.covers("skanetrafiken", _leg("Öresundståg", SK_MALMO, SK_LUND))
    assert cov.covers("skanetrafiken", _leg("Öresundståg", SK_LUND, SK_OSBY))
    assert cov.covers("skanetrafiken", _leg("Öresundståg", SK_MALMO, SK_BASTAD))


def test_extra_region_stops_restore_a_station_the_pta_buses_never_serve():
    """Hallandstrafiken's own feed vehicles are buses; every train at Halmstad C runs under a
    train operator's agency, so the county's MAIN station is absent from the agency-derived
    region. extra_region_stops puts it back as a full region member - single-card and
    combined coverage alike - because the card genuinely covers it."""
    from tripps.passes import load_cards

    card = load_cards()["hallandstrafiken"]
    assert "740000080" in card.extra_region_stops

    laholm = Stop(id="740000058", name="Laholm", lat=56.502, lon=13.000)
    halmstad = Stop(id="740000080", name="Halmstad C", lat=56.669, lon=12.865)
    b = TimetableBuilder()
    b.add_stop(laholm)
    b.add_stop(halmstad)
    b.add_trip(
        RouteInfo(id="ore", mode=TransportMode.TRAIN, operator="Öresundståg"),
        [laholm.id, halmstad.id],
        Trip(id="o1", arrivals=[0, 600], departures=[0, 600]),
    )
    cov = PassCoverage(
        b.build(),
        cards={"hallandstrafiken": card},
        # The agency's own (bus) reach includes Laholm's parent but NOT the Halmstad rail one.
        agency_stops={"Hallandstrafiken": [laholm.id]},
    )
    assert cov.covers("hallandstrafiken", _leg("Öresundståg", laholm, halmstad))


# --- combined coverage must not pool border stops -----------------------------


def test_combined_border_stops_do_not_bridge_a_gap_between_two_cards():
    """SC lies in NEITHER card's region; both merely name it as a single-card border
    extension. Pooling border stops as covered let the pair 'jointly' free SA->SD across a
    hop neither card covers - the union must be region stops only."""
    reg1 = TravelCard(
        id="reg1", name="Region One", region="R1",
        honored_operators=frozenset({"Kust"}), coverage_model="region-stops",
        region_agencies=frozenset({"Ag1"}), border_stops=frozenset({"SC"}),
    )
    reg2 = TravelCard(
        id="reg2", name="Region Two", region="R2",
        honored_operators=frozenset({"Kust"}), coverage_model="region-stops",
        region_agencies=frozenset({"Ag2"}), border_stops=frozenset({"SC"}),
    )
    b = TimetableBuilder()
    for s in (SA, SB, SC, SD):
        b.add_stop(s)
    b.add_trip(
        RouteInfo(id="k", mode=TransportMode.TRAIN, operator="Kust"),
        ["SA", "SB", "SC", "SD"],
        Trip(id="k1", arrivals=[0, 6, 12, 18], departures=[0, 6, 12, 18]),
    )
    cov = PassCoverage(
        b.build(),
        cards={"reg1": reg1, "reg2": reg2},
        agency_stops={"Ag1": ["SA", "SB"], "Ag2": ["SD"]},
    )
    # The gap hop SB->SC->SD is bridged only by border stops -> NOT combined-covered.
    assert cov.combined_cover(["reg1", "reg2"], _through(SA, SD, ["SA", "SB", "SC", "SD"])) is None
    # The single-card directional border extension is untouched: SB (region) -> SC (border).
    assert cov.covers("reg1", _leg("Kust", SB, SC))


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


# --- web-verified cross-county curation (2026-07 geography audit) -------------

def test_curated_cards_never_free_verified_invalid_stations():
    """Each PTA's own vehicles cross its county line, putting stations in the agency stop set
    where the county card is NOT valid (web-verified per PTA). covers() must refuse them all,
    while an in-county leg on the same card stays covered."""
    from tripps.passes import load_cards

    # card -> (home agency, honored operator to test with,
    #          one in-county station id kept covered, [out-of-county denied ids])
    cases = {
        "kalmar-lanstrafik": ("Kalmar Länstrafik", "Krösatåg", "740000075",
                              ["740000009", "740000230", "740000140"]),
        "jonkopings-lanstrafik": ("Jönköpings Länstrafik", "Krösatåg", "740000140",
                                  ["740000080", "740000250", "740000348"]),
        "vasttrafik": ("Västtrafik", "Västtrafik", "740000002",
                       ["740000133", "740000077", "740000140"]),
        "ostgotatrafiken": ("Östgötatrafiken", "Östgötatrafiken", "740000009",
                            ["740000084"]),
        "din-tur": ("Din Tur", "Din Tur", "740000130",
                    ["740000434"]),
        "lanstrafiken-jamtland": ("Länstrafiken Jämtland", "Norrtåg", "740000123",
                                  ["740001570", "740000302"]),
        "lanstrafiken-vasterbotten": ("Länstrafiken Västerbotten", "Norrtåg", "740000175",
                                      ["740000123", "740000254"]),
        "lanstrafiken-kronoberg": ("Länstrafiken Kronoberg", "Öresundståg", "740000250",
                                   ["740043449"]),
    }
    cards = load_cards()
    for cid, (agency, operator, home_id, denied_ids) in cases.items():
        stops = {sid: Stop(id=sid, name=sid, lat=60.0, lon=15.0)
                 for sid in [home_id, *denied_ids, "HOME2"]}
        b = TimetableBuilder()
        for s in stops.values():
            b.add_stop(s)
        b.add_trip(
            RouteInfo(id=f"r-{cid}", mode=TransportMode.TRAIN, operator=operator),
            list(stops), Trip(id="t", arrivals=list(range(0, 6 * len(stops), 6)),
                              departures=list(range(0, 6 * len(stops), 6))),
        )
        cov = PassCoverage(
            b.build(), cards={cid: cards[cid]},
            agency_stops={agency: list(stops)},  # the agency's reach includes them ALL
        )
        home, home2 = stops[home_id], stops["HOME2"]
        assert cov.covers(cid, _leg(operator, home, home2)), f"{cid}: in-county must stay covered"
        for did in denied_ids:
            assert not cov.covers(cid, _leg(operator, home, stops[did])), (
                f"{cid}: {did} is web-verified INVALID and must not be freed"
            )


def test_x_trafik_verified_valid_extensions_stay_covered():
    """The audit confirmed x-trafik's Sundsvall + Falun coverage is genuine (cooperation
    agreements) - the curation must NOT have denied them."""
    from tripps.passes import load_cards

    card = load_cards()["x-trafik"]
    for sid in ("740000130", "740020045", "740001622", "740000030"):
        assert sid not in card.denied_stops
