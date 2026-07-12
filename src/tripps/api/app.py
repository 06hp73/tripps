"""FastAPI service and web UI.

The planner needs a timetable in memory, so the app builds one at startup from the GTFS
zip and rebuilds it when the service date rolls over. Freerider inventory is polled in the
background, because a car listed five minutes ago may already be gone.

Every endpoint answers even when a price source is down: itineraries come back with the
legs it could price, warnings for the ones it could not, and per-source health so the UI can
say "SJ prices unavailable" instead of pretending the trip is cheap.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from ..calibration import load_calibrated_floors
from ..config import Settings, get_settings
from ..db import Database
from ..ingest.flights import GoogleFlightsProvider, NullFlightProvider
from ..ingest.freerider import (
    FreeriderClient,
    FreeriderCostModel,
    offers_to_log_rows,
    only_within,
    parse_offers,
    schema_drift,
)
from ..ingest.gtfs import GtfsConfig, extract_agency_stops, load_timetable_cached
from ..models import SearchConstraints, TransportMode
from ..passes import (
    PassAdapter,
    PassCoverage,
    TickitalAdapter,
    TickitalRental,
    load_cards,
)
from ..pricing.flights import FlightAdapter
from ..pricing.flixbus import FlixBusAdapter
from ..pricing.freerider import FreeriderAdapter
from ..pricing.operators import DeeplinkAdapter, StaticFareAdapter
from ..pricing.orchestrator import PricingOrchestrator
from ..pricing.sj import SJAdapter
from ..pricing.tora import ToraAdapter
from ..routing.timetable import Timetable
from ..search import (
    Planner,
    SearchOptions,
    cheapest_over_window,
    round_trip,
    summarize,
    window_rental_notes,
)
from ..timeutil import now_local
from ..watcher import (
    DEFAULT_WATCH_RADIUS_KM,
    Watch,
    hit_payload,
    match_watches,
    notify_webhook,
)

log = logging.getLogger(__name__)

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class AppState:
    """Holds the things that are expensive to build and cheap to share."""

    #: How many days' timetables to keep in memory. Loading one costs ~15 s of parsing the
    #: 564 MB feed, so a user flipping between a few dates should pay it once each, not every
    #: time. Each built timetable is tens of MB, so a handful is cheap to retain.
    TIMETABLE_CACHE_SIZE = 6

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.db_path)
        self.orchestrator: PricingOrchestrator | None = None
        self.flight_provider = None
        self._timetables: OrderedDict[date, Timetable] = OrderedDict()
        self.timetable_date: date | None = None
        self.freerider_offers: list = []
        self.freerider_client = FreeriderClient(
            base_url=settings.freerider_base, user_agent=settings.user_agent
        )
        self.errors: list[str] = []
        self._poller: asyncio.Task | None = None

    # --- timetable --------------------------------------------------------

    def _ensure_orchestrator(self) -> PricingOrchestrator:
        """Build the pricing adapters once and share them across dates.

        The adapters are date-independent and hold live state that must persist across
        searches - the SJ subscription key, FlixBus's city and search caches, per-source
        health. Rebuilding them per date would throw all of that away and re-warm it every
        time the user changed the date.
        """
        if self.orchestrator is not None:
            return self.orchestrator
        self.flight_provider = (
            GoogleFlightsProvider() if _fast_flights_available() else NullFlightProvider()
        )
        # Full-feed agency->stops map defines each travel card's region (see passes.py); the
        # county PTAs whose only routes are local buses are absent from the routing timetable,
        # so their regions must come from the whole feed. Disk-cached; loaded once here.
        agency_stops = None
        if self.settings.gtfs_zip_path.exists():
            agency_stops = extract_agency_stops(
                self.settings.gtfs_zip_path, self.settings.data_dir / "tt-cache"
            )
        adapters = [
            # First: a held travel card zeroes a covered leg before any paid source is asked.
            PassAdapter(agency_stops=agency_stops),
            FlixBusAdapter(
                base_url=self.settings.flixbus_base,
                user_agent=self.settings.user_agent,
                timeout=self.settings.http_timeout_seconds,
                min_interval=self.settings.budget.min_interval_seconds,
                db=self.db,
            ),
            SJAdapter(
                user_agent=self.settings.user_agent,
                timeout=self.settings.http_timeout_seconds,
            ),
            ToraAdapter(
                user_agent=self.settings.user_agent,
                min_interval=self.settings.budget.min_interval_seconds,
            ),
            FreeriderAdapter(cost_model=FreeriderCostModel()),
            FlightAdapter(),
            _fare_table(self.settings),
            # Second-to-last: a rental prices only a covered leg no paid source could, so the
            # itinerary coupon still has real singles to compare against.
            TickitalAdapter(agency_stops=agency_stops),
            # Last: anything a real price source could not answer becomes a booking link.
            DeeplinkAdapter(),
        ]
        self.orchestrator = PricingOrchestrator(
            adapters,
            self.db,
            budget=self.settings.budget,
            ttl=self.settings.ttl,
            # Floors calibrated from logged fares if any exist, else the safe defaults.
            floors=load_calibrated_floors(self.db),
        )
        return self.orchestrator

    def _timetable_for(self, service_date: date) -> Timetable:
        cached = self._timetables.get(service_date)
        if cached is not None:
            self._timetables.move_to_end(service_date)
            return cached

        zip_path = self.settings.gtfs_zip_path
        if not zip_path.exists():
            raise FileNotFoundError(f"no GTFS feed at {zip_path}. Run `tripps fetch-gtfs` first.")
        # Disk-cached: a date is parsed once (~15 s) and thereafter unpickled (<1 s), which is
        # what makes date-range ("cheapest day") searches and cold-date searches usable.
        timetable = load_timetable_cached(
            zip_path, service_date, GtfsConfig(), cache_dir=self.settings.data_dir / "tt-cache"
        )
        log.info(
            "timetable for %s: %d stops, %d trips",
            service_date,
            timetable.num_stops,
            timetable.num_trips,
        )

        self._timetables[service_date] = timetable
        self._timetables.move_to_end(service_date)
        while len(self._timetables) > self.TIMETABLE_CACHE_SIZE:
            self._timetables.popitem(last=False)
        return timetable

    @property
    def current_timetable(self) -> Timetable | None:
        """The most recently used timetable, for read-only introspection (health, stops)."""
        if self.timetable_date in self._timetables:
            return self._timetables[self.timetable_date]
        if self._timetables:
            return next(reversed(self._timetables.values()))
        return None

    def planner_for(self, service_date: date) -> Planner:
        """A planner for one service date, reusing the shared adapters and cached timetable.

        A GTFS feed encodes one calendar, so a query for tomorrow needs tomorrow's active
        services; but the pricing adapters do not depend on the date, so only the timetable
        varies. The timetable is cached per date (see `_timetable_for`) so switching dates
        does not re-parse the feed each time.
        """
        timetable = self._timetable_for(service_date)
        self.timetable_date = service_date
        orchestrator = self._ensure_orchestrator()
        return Planner(
            timetable,
            orchestrator,
            floors=orchestrator.floors,  # the router prunes with the same calibrated floors
            db=self.db,
            options=SearchOptions(),
            flight_provider=self.flight_provider,
        )

    # --- freerider polling ------------------------------------------------

    async def refresh_freerider(self) -> None:
        try:
            raw = await self.freerider_client.fetch_raw("SWEDEN")
            offers = only_within(parse_offers(raw), "se")
        except Exception as exc:  # noqa: BLE001 - an undocumented endpoint may vanish
            log.warning("freerider refresh failed: %s", exc)
            self.db.set_health("freerider-inventory", "down", str(exc))
            cached = self.db.latest_freerider_snapshot()
            if cached and not self.freerider_offers:
                self.freerider_offers = only_within(parse_offers([{"routes": cached}]), "se")
            return

        drift = schema_drift(offers)
        if drift:
            # The meaning of `distance` underpins the whole Freerider cost model.
            log.error("freerider schema drift: %s", drift[:3])
            self.db.set_health("freerider-inventory", "degraded", drift[0])
        else:
            self.db.set_health("freerider-inventory", "ok", f"{len(offers)} offers")

        self.freerider_offers = offers
        self.db.log_freerider_offers(offers_to_log_rows(offers, raw))
        log.info("freerider: %d offers within Sweden", len(offers))
        await self.check_watches()

    async def check_watches(self) -> int:
        """Match the current inventory against active watches, announcing new cars once.

        Runs after every inventory refresh. A car is announced the first time it matches a
        watch (record_hit dedupes on (watch, car)), by webhook if one is configured and always
        into the hit log the UI and CLI read.
        """
        watches = [Watch.from_row(r) for r in self.db.list_watches(active_only=True)]
        if not watches:
            return 0
        new_hits = 0
        for watch, offer in match_watches(watches, self.freerider_offers):
            is_new = self.db.record_hit(
                watch.id,
                route_id=offer.route_id,
                pickup=offer.pickup.name,
                dropoff=offer.dropoff.name,
                available_at=offer.available_at.isoformat(),
                car=offer.car_model,
            )
            if not is_new:
                continue
            new_hits += 1
            payload = hit_payload(watch, offer)
            log.info(
                "freerider watch %d matched: %s -> %s (%s)",
                watch.id, offer.pickup.name, offer.dropoff.name, offer.car_model,
            )
            if watch.webhook_url:
                await notify_webhook(watch.webhook_url, payload)
        return new_hits

    async def warm_sj_key(self) -> None:
        """Resolve the SJ booking key ahead of the first search, tolerating failure."""
        if self.orchestrator is None:
            return
        for adapter in self.orchestrator.adapters:
            ensure = getattr(adapter, "ensure_key", None)
            if ensure is None:
                continue
            try:
                await ensure()
            except Exception as exc:  # noqa: BLE001 - the app must still serve
                log.warning("SJ key warm-up failed: %s", exc)

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.freerider_poll_seconds)
            await self.refresh_freerider()

    def start_polling(self) -> None:
        """Start the background refresh. The *first* fetch happens in `lifespan`, before the
        app serves anything, so the very first search does not silently see zero cars."""
        self._poller = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._poller is not None:
            self._poller.cancel()
            try:
                await self._poller
            except asyncio.CancelledError:
                pass
        await self.freerider_client.aclose()
        if self.orchestrator is not None:
            for adapter in self.orchestrator.adapters:
                await adapter.aclose()
        self.db.close()


def _fast_flights_available() -> bool:
    try:
        import fast_flights  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _fare_table(settings: Settings) -> StaticFareAdapter:
    """Load `data/fares.json` when present. Absent by default, and that is deliberate."""
    path = settings.data_dir / "fares.json"
    if path.exists():
        return StaticFareAdapter.from_file(path)
    return StaticFareAdapter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    state = AppState(settings)
    app.state.app_state = state

    try:
        state.planner_for(now_local().date())
    except FileNotFoundError as exc:
        # The app must still start so /health can explain what is missing.
        log.error("%s", exc)
        state.errors.append(str(exc))

    # Load the car inventory before serving: a search that runs in the gap before the first
    # poll would report "no Freerider offers" and quietly drop the cheapest itineraries.
    await state.refresh_freerider()
    # Resolve the SJ subscription key now, so the first search does not pay for extracting
    # it from the site bundle mid-request.
    await state.warm_sj_key()
    state.start_polling()
    try:
        yield
    finally:
        await state.stop()


app = FastAPI(title="tripps", version="0.1.0", lifespan=lifespan)


def _state(request: Request) -> AppState:
    return request.app.state.app_state


def _parse_date(value: str | None) -> date:
    if not value:
        return now_local().date()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(400, f"bad date {value!r}, expected YYYY-MM-DD") from exc


def _constraints(
    max_hours: float | None,
    max_transfers: int | None,
    include_freerider: bool,
    modes: list[str] | None,
    earliest: str | None,
    service_date: date,
) -> SearchConstraints:
    if modes:
        try:
            allowed = frozenset(TransportMode(m) for m in modes) | {TransportMode.WALK}
        except ValueError as exc:
            raise HTTPException(400, f"unknown mode: {exc}") from exc
    else:
        # Flights are opt-in: each domestic route is an ~8-second Google Flights scrape, so a
        # default search that silently included them would always feel slow. A caller enables
        # them explicitly with modes=flight.
        allowed = frozenset(TransportMode) - {TransportMode.FLIGHT}

    earliest_dt = None
    if earliest:
        try:
            hour, minute = (int(p) for p in earliest.split(":", 1))
            earliest_dt = datetime.combine(
                service_date, datetime.min.time().replace(hour=hour, minute=minute)
            ).replace(tzinfo=now_local().tzinfo)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"bad time {earliest!r}, expected HH:MM") from exc

    return SearchConstraints(
        max_duration_seconds=int(max_hours * 3600) if max_hours else None,
        max_transfers=max_transfers,
        include_freerider=include_freerider,
        allowed_modes=allowed,
        earliest_departure=earliest_dt,
    )


def _rentals(db) -> list[TickitalRental]:
    """All registered tickital rentals, as pass objects the planner understands."""
    return [TickitalRental.from_row(r) for r in db.list_tickital_rentals()]


async def _run_search(state: AppState, origin, destination, service_date, constraints):
    planner = state.planner_for(service_date)
    try:
        return await planner.search(
            origin,
            destination,
            service_date,
            constraints=constraints,
            offers=state.freerider_offers,
            held_cards=state.db.list_cards(),
            tickital_rentals=_rentals(state.db),
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


# --- JSON API --------------------------------------------------------------


@app.get("/api/search")
async def api_search(
    request: Request,
    origin: str = Query(..., min_length=2),
    destination: str = Query(..., min_length=2),
    date_: str | None = Query(None, alias="date"),
    max_hours: float | None = None,
    max_transfers: int | None = None,
    include_freerider: bool = True,
    earliest: str | None = None,
    modes: list[str] | None = Query(None),
) -> JSONResponse:
    state = _state(request)
    service_date = _parse_date(date_)
    constraints = _constraints(
        max_hours, max_transfers, include_freerider, modes, earliest, service_date
    )
    response, stats = await _run_search(state, origin, destination, service_date, constraints)
    payload = response.model_dump(mode="json")
    payload["stats"] = {
        "candidates": stats.candidates,
        "priced": stats.priced,
        "upstream_calls": stats.upstream_calls,
        "freerider_offers": stats.freerider_offers,
        "flight_offers": stats.flight_offers,
        "rounds": stats.rounds,
    }
    return JSONResponse(payload)


def _make_planner(state: AppState):
    def factory(service_date: date):
        return state.planner_for(service_date)

    return factory


@app.get("/api/round-trip")
async def api_round_trip(
    request: Request,
    origin: str = Query(..., min_length=2),
    destination: str = Query(..., min_length=2),
    outbound: str = Query(..., alias="date"),
    return_date: str = Query(..., alias="return"),
    max_hours: float | None = None,
    max_transfers: int | None = None,
    include_freerider: bool = True,
    modes: list[str] | None = Query(None),
) -> JSONResponse:
    state = _state(request)
    out_date, back_date = _parse_date(outbound), _parse_date(return_date)
    constraints = _constraints(
        max_hours, max_transfers, include_freerider, modes, None, out_date
    )
    try:
        result = await round_trip(
            _make_planner(state), origin, destination, out_date, back_date,
            constraints=constraints, offers=state.freerider_offers,
            held_cards=state.db.list_cards(), tickital_rentals=_rentals(state.db),
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc

    return JSONResponse(
        {
            "outbound": result.outbound.model_dump(mode="json"),
            "inbound": result.inbound.model_dump(mode="json"),
            "total_price_sek": (result.total_price_ore / 100) if result.total_price_ore else None,
            "warnings": result.warnings,
        }
    )


@app.get("/api/fares")
async def api_fares(
    request: Request,
    origin: str = Query(..., min_length=2),
    destination: str = Query(..., min_length=2),
    start: str | None = Query(None),
    days: int = Query(7, ge=1, le=14),
    max_hours: float | None = None,
    max_transfers: int | None = None,
    include_freerider: bool = True,
    modes: list[str] | None = Query(None),
) -> JSONResponse:
    """Cheapest priced itinerary per day over a window - a fare calendar."""
    state = _state(request)
    start_date = _parse_date(start)
    constraints = _constraints(
        max_hours, max_transfers, include_freerider, modes, None, start_date
    )
    rentals = _rentals(state.db)
    try:
        fares = await cheapest_over_window(
            _make_planner(state), origin, destination, start_date, days,
            constraints=constraints, offers=state.freerider_offers,
            held_cards=state.db.list_cards(), tickital_rentals=rentals,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc

    return JSONResponse(
        {
            "origin": origin,
            "destination": destination,
            "notes": window_rental_notes(fares, rentals),
            "days": [
                {
                    "date": f.date.isoformat(),
                    "price_sek": f.price_sek,
                    "modes": [
                        m.value for m in f.cheapest.modes if m.value != "walk"
                    ] if f.cheapest else [],
                    "departure": f.cheapest.departure.isoformat() if f.cheapest else None,
                    # The full cheapest itinerary, so pressing a day can expand its legs and
                    # booking links without a second request.
                    "itinerary": f.cheapest.model_dump(mode="json") if f.cheapest else None,
                }
                for f in fares
            ],
            "cheapest": min(
                (
                    {"date": f.date.isoformat(), "price_sek": f.price_sek}
                    for f in fares
                    if f.price_ore is not None
                ),
                key=lambda d: d["price_sek"],
                default=None,
            ),
        }
    )


@app.get("/api/stops")
async def api_stops(request: Request, q: str = Query(..., min_length=2)) -> JSONResponse:
    state = _state(request)
    try:
        planner = state.planner_for(state.timetable_date or now_local().date())
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return JSONResponse(
        [{"id": s.id, "name": s.name} for s in planner.resolve_stops(q, limit=8)]
    )


@app.get("/api/freerider")
async def api_freerider(request: Request) -> JSONResponse:
    state = _state(request)
    cost = FreeriderCostModel()
    return JSONResponse(
        [
            {
                "route_id": o.route_id,
                "from": o.pickup.name,
                "to": o.dropoff.name,
                "available_at": o.available_at.isoformat(),
                "latest_return": o.latest_return.isoformat(),
                "car": o.car_model,
                "direct_km": o.direct_km,
                "included_km": o.included_km,
                "estimated_cost_sek": cost.estimate_ore(o) / 100,
            }
            for o in state.freerider_offers
        ]
    )


@app.get("/api/providers")
async def api_providers(request: Request) -> JSONResponse:
    """Every regional travel-card provider, for the searchable scroll-list.

    `supported` marks whether the card can actually free legs on the current network (its home
    region resolved to stops); unsupported cards are still listed so the user sees their region.
    """
    state = _state(request)
    coverage = None
    if state.current_timetable is not None:
        agency_stops = (
            extract_agency_stops(state.settings.gtfs_zip_path, state.settings.data_dir / "tt-cache")
            if state.settings.gtfs_zip_path.exists()
            else None
        )
        coverage = PassCoverage(state.current_timetable, agency_stops=agency_stops)
    # held covers full + partial holdings; partial (all_zone=0) cards are marked so the UI can
    # show they are registered but not auto-freed.
    held = {r["provider_id"]: bool(r["all_zone"]) for r in state.db.list_card_rows()}
    return JSONResponse(
        {
            "providers": [
                {
                    "id": c.id,
                    "name": c.name,
                    "region": c.region,
                    "held": c.id in held,
                    "all_zone": held.get(c.id, True),
                    "supported": coverage.is_supported(c.id) if coverage else True,
                    "note": c.coverage_note,
                }
                for c in sorted(load_cards().values(), key=lambda x: x.name)
            ]
        }
    )


@app.get("/api/cards")
async def api_cards_list(request: Request) -> JSONResponse:
    state = _state(request)
    cards = load_cards()
    return JSONResponse(
        {
            "cards": [
                {
                    "id": r["provider_id"],
                    "name": cards[r["provider_id"]].name if r["provider_id"] in cards else r["provider_id"],
                    "all_zone": bool(r["all_zone"]),
                }
                for r in state.db.list_card_rows()
            ]
        }
    )


@app.post("/api/cards")
async def api_cards_add(request: Request) -> JSONResponse:
    state = _state(request)
    body = await request.json()
    provider_id = body.get("provider_id")
    if provider_id not in load_cards():
        raise HTTPException(400, f"unknown provider: {provider_id}")
    # A partial-zone holding is registered but not auto-freed - its exact zones cannot be
    # verified from feed data, so its legs are priced as paid rather than wrongly zeroed.
    all_zone = bool(body.get("all_zone", True))
    state.db.add_card(provider_id, all_zone=all_zone)
    resp = {"added": provider_id, "all_zone": all_zone}
    if not all_zone:
        resp["note"] = (
            "Registered as a partial-zone ticket: its legs are priced as paid, not auto-freed, "
            "because per-zone coverage can't be verified from the timetable data."
        )
    return JSONResponse(resp)


@app.delete("/api/cards/{provider_id}")
async def api_cards_remove(request: Request, provider_id: str) -> JSONResponse:
    state = _state(request)
    if not state.db.remove_card(provider_id):
        raise HTTPException(404, f"card not registered: {provider_id}")
    return JSONResponse({"removed": provider_id})


def _rental_json(row) -> dict:
    cards = load_cards()
    card = cards.get(row["provider_id"])
    return {
        "id": row["id"],
        "provider_id": row["provider_id"],
        "provider_name": card.name if card else row["provider_id"],
        "price_sek": row["price_ore"] / 100,
        "valid_from": row["valid_from"],
        "valid_to": row["valid_to"],
        "note": row["note"] or "",
    }


@app.get("/api/rentals")
async def api_rentals_list(request: Request) -> JSONResponse:
    state = _state(request)
    return JSONResponse(
        {"rentals": [_rental_json(r) for r in state.db.list_tickital_rentals()]}
    )


@app.post("/api/rentals")
async def api_rentals_add(request: Request) -> JSONResponse:
    """Register a second-hand tickital rental: a provider's card, priced, over a date window.

    Registered by hand from what the user sees in the tickital app (it has no scrapable
    listings surface). The response carries the terms-of-service warning so it is never a
    silent side effect.
    """
    state = _state(request)
    body = await request.json()
    provider_id = body.get("provider_id")
    if provider_id not in load_cards():
        raise HTTPException(400, f"unknown provider: {provider_id}")
    try:
        price_ore = round(float(body["price_sek"]) * 100)
        valid_from = date.fromisoformat(body["valid_from"])
        valid_to = date.fromisoformat(body["valid_to"])
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(400, f"bad rental: {exc}") from exc
    if price_ore < 0:
        raise HTTPException(400, "price must not be negative")
    if valid_to < valid_from:
        raise HTTPException(400, "valid_to is before valid_from")
    rental_id = state.db.add_tickital_rental(
        provider_id=provider_id,
        price_ore=price_ore,
        valid_from=valid_from,
        valid_to=valid_to,
        note=str(body.get("note", "")),
    )
    return JSONResponse(
        {
            "added": rental_id,
            "warning": (
                f"Renting a period ticket violates {load_cards()[provider_id].name}'s terms; "
                "the ticket can be blocked. Registered for your own comparison only."
            ),
        }
    )


@app.delete("/api/rentals/{rental_id}")
async def api_rentals_remove(request: Request, rental_id: int) -> JSONResponse:
    state = _state(request)
    if not state.db.remove_tickital_rental(rental_id):
        raise HTTPException(404, f"rental not found: {rental_id}")
    return JSONResponse({"removed": rental_id})


@app.post("/api/watch")
async def api_watch_create(request: Request) -> JSONResponse:
    """Register a Freerider route to watch. Matches are announced as inventory arrives."""
    state = _state(request)
    body = await request.json()
    origin_q, dest_q = body.get("origin"), body.get("destination")
    if not origin_q or not dest_q:
        raise HTTPException(400, "origin and destination are required")

    try:
        planner = state.planner_for(state.timetable_date or now_local().date())
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    origins = planner.resolve_stops(origin_q, limit=1)
    dests = planner.resolve_stops(dest_q, limit=1)
    if not origins or not dests:
        raise HTTPException(404, "could not resolve origin or destination")
    origin, dest = origins[0], dests[0]

    watch_id = state.db.add_watch(
        origin=origin.name,
        destination=dest.name,
        origin_lat=origin.lat,
        origin_lon=origin.lon,
        dest_lat=dest.lat,
        dest_lon=dest.lon,
        radius_km=float(body.get("radius_km") or DEFAULT_WATCH_RADIUS_KM),
        webhook_url=body.get("webhook_url"),
    )
    # Check the current inventory immediately, so a car that already exists is not missed.
    await state.check_watches()
    return JSONResponse({"id": watch_id, "origin": origin.name, "destination": dest.name})


@app.get("/api/watch")
async def api_watch_list(request: Request) -> JSONResponse:
    state = _state(request)
    return JSONResponse(
        {
            "watches": [
                {
                    "id": w["id"],
                    "origin": w["origin"],
                    "destination": w["destination"],
                    "radius_km": w["radius_km"],
                }
                for w in state.db.list_watches(active_only=True)
            ],
            "recent_hits": [
                {
                    "watch_id": h["watch_id"],
                    "pickup": h["pickup"],
                    "dropoff": h["dropoff"],
                    "available_at": h["available_at"],
                    "car": h["car"],
                    "seen_at": h["seen_at"],
                }
                for h in state.db.recent_hits(limit=50)
            ],
        }
    )


@app.delete("/api/watch/{watch_id}")
async def api_watch_delete(request: Request, watch_id: int) -> JSONResponse:
    state = _state(request)
    if not state.db.deactivate_watch(watch_id):
        raise HTTPException(404, f"no watch {watch_id}")
    return JSONResponse({"deactivated": watch_id})


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    state = _state(request)
    violations = state.db.floor_violations()
    all_health = state.db.get_health()
    # Split the scheduled-canary rows out from the live per-request source health.
    canaries = {k.removeprefix("canary:"): v for k, v in all_health.items() if k.startswith("canary:")}
    sources = {k: v for k, v in all_health.items() if not k.startswith("canary:")}
    return JSONResponse(
        {
            "status": "ok" if not state.errors else "degraded",
            "errors": state.errors,
            "timetable_date": state.timetable_date.isoformat() if state.timetable_date else None,
            "timetable_dates_cached": [d.isoformat() for d in state._timetables],
            "stops": state.current_timetable.num_stops if state.current_timetable else 0,
            "trips": state.current_timetable.num_trips if state.current_timetable else 0,
            "freerider_offers": len(state.freerider_offers),
            "sources": sources,
            "canaries": canaries,
            "flight_scraper": "available" if _fast_flights_available() else "not installed",
            # A floor above a real fare means the router may have pruned the cheapest trip.
            "price_floor_violations": len(violations),
        }
    )


@app.get("/api/canary")
async def api_canary(request: Request) -> JSONResponse:
    """Probe every live price source now. Slow (hits real endpoints); for monitoring."""
    from ..monitoring import persist_canaries, run_canaries

    state = _state(request)
    results = await run_canaries(state.settings)
    persist_canaries(state.db, results)
    worst = "down" if any(not r.ok for r in results) else "ok"
    return JSONResponse(
        {
            "status": worst,
            "results": [
                {"name": r.name, "state": r.state.value, "detail": r.detail, "latency_ms": r.latency_ms}
                for r in results
            ],
        }
    )


# --- web UI ----------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    state = _state(request)
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "today": now_local().date().isoformat(),
            "freerider_count": len(state.freerider_offers),
            "errors": state.errors,
        },
    )


@app.post("/search", response_class=HTMLResponse)
async def ui_search(
    request: Request,
    origin: str = Form(...),
    destination: str = Form(...),
    date_: str = Form(alias="date"),
    max_hours: float | None = Form(None),
    max_transfers: int | None = Form(None),
    earliest: str | None = Form(None),
    mode_train: bool = Form(False),
    mode_bus: bool = Form(False),
    mode_ferry: bool = Form(False),
    mode_freerider: bool = Form(False),
    mode_flight: bool = Form(False),
) -> HTMLResponse:
    state = _state(request)
    service_date = _parse_date(date_)

    # local_transit is a connective mode (feeder legs), always allowed; the checkboxes
    # choose the long-distance modes.
    modes = ["local_transit"]
    for enabled, name in (
        (mode_train, "train"),
        (mode_bus, "bus"),
        (mode_ferry, "ferry"),
        (mode_freerider, "freerider"),
        (mode_flight, "flight"),
    ):
        if enabled:
            modes.append(name)

    constraints = _constraints(
        max_hours, max_transfers, mode_freerider, modes, earliest, service_date
    )
    try:
        response, stats = await _run_search(
            state, origin, destination, service_date, constraints
        )
    except HTTPException as exc:
        return TEMPLATES.TemplateResponse(
            request, "_results.html", {"error": exc.detail, "itineraries": []}, status_code=200
        )

    return TEMPLATES.TemplateResponse(
        request,
        "_results.html",
        {
            "response": response,
            "itineraries": response.itineraries,
            "stats": stats,
            "summarize": summarize,
            "error": None,
        },
    )
