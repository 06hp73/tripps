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
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

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
from ..ingest.gtfs import GtfsConfig, load_timetable
from ..models import SearchConstraints, TransportMode
from ..pricing.flights import FlightAdapter
from ..pricing.flixbus import FlixBusAdapter
from ..pricing.freerider import FreeriderAdapter
from ..pricing.operators import DeeplinkAdapter, StaticFareAdapter
from ..pricing.orchestrator import PricingOrchestrator
from ..routing.floors import PriceFloorModel
from ..search import Planner, SearchOptions, summarize
from ..timeutil import now_local

log = logging.getLogger(__name__)

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class AppState:
    """Holds the things that are expensive to build and cheap to share."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.db_path)
        self.planner: Planner | None = None
        self.timetable_date: date | None = None
        self.freerider_offers: list = []
        self.freerider_client = FreeriderClient(
            base_url=settings.freerider_base, user_agent=settings.user_agent
        )
        self.errors: list[str] = []
        self._poller: asyncio.Task | None = None

    # --- timetable --------------------------------------------------------

    def build_planner(self, service_date: date) -> Planner:
        zip_path = self.settings.gtfs_zip_path
        if not zip_path.exists():
            raise FileNotFoundError(f"no GTFS feed at {zip_path}. Run `tripps fetch-gtfs` first.")
        timetable, stats = load_timetable(zip_path, service_date, GtfsConfig())
        log.info(
            "timetable for %s: %d stops, %d routes, %d trips",
            service_date,
            stats.stops,
            stats.routes,
            stats.trips,
        )
        if stats.problems:
            log.warning("gtfs problems: %s", stats.problems[:5])

        flight_provider = (
            GoogleFlightsProvider() if _fast_flights_available() else NullFlightProvider()
        )
        adapters = [
            FlixBusAdapter(
                base_url=self.settings.flixbus_base,
                user_agent=self.settings.user_agent,
                timeout=self.settings.http_timeout_seconds,
                min_interval=self.settings.budget.min_interval_seconds,
                db=self.db,
            ),
            FreeriderAdapter(cost_model=FreeriderCostModel()),
            FlightAdapter(),
            _fare_table(self.settings),
            # Last: anything a real price source could not answer becomes a booking link.
            DeeplinkAdapter(),
        ]
        orchestrator = PricingOrchestrator(
            adapters,
            self.db,
            budget=self.settings.budget,
            ttl=self.settings.ttl,
            floors=PriceFloorModel(),
        )
        self.timetable_date = service_date
        return Planner(
            timetable,
            orchestrator,
            db=self.db,
            options=SearchOptions(),
            flight_provider=flight_provider,
        )

    def planner_for(self, service_date: date) -> Planner:
        """Rebuild the timetable when the requested service date changes.

        A GTFS feed encodes one calendar; a query for tomorrow needs tomorrow's active
        services, not today's.
        """
        if self.planner is None or self.timetable_date != service_date:
            self.planner = self.build_planner(service_date)
        return self.planner

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
        if self.planner is not None:
            for adapter in self.planner.orchestrator.adapters:
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
    allowed = frozenset(TransportMode)
    if modes:
        try:
            allowed = frozenset(TransportMode(m) for m in modes) | {TransportMode.WALK}
        except ValueError as exc:
            raise HTTPException(400, f"unknown mode: {exc}") from exc

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


async def _run_search(state: AppState, origin, destination, service_date, constraints):
    planner = state.planner_for(service_date)
    try:
        return await planner.search(
            origin,
            destination,
            service_date,
            constraints=constraints,
            offers=state.freerider_offers,
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


@app.get("/api/stops")
async def api_stops(request: Request, q: str = Query(..., min_length=2)) -> JSONResponse:
    state = _state(request)
    planner = state.planner
    if planner is None:
        raise HTTPException(503, "timetable not loaded")
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


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    state = _state(request)
    violations = state.db.floor_violations()
    return JSONResponse(
        {
            "status": "ok" if not state.errors else "degraded",
            "errors": state.errors,
            "timetable_date": state.timetable_date.isoformat() if state.timetable_date else None,
            "stops": state.planner.timetable.num_stops if state.planner else 0,
            "trips": state.planner.timetable.num_trips if state.planner else 0,
            "freerider_offers": len(state.freerider_offers),
            "sources": state.db.get_health(),
            "flight_scraper": "available" if _fast_flights_available() else "not installed",
            # A floor above a real fare means the router may have pruned the cheapest trip.
            "price_floor_violations": len(violations),
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
    include_freerider: bool = Form(False),
    include_flights: bool = Form(False),
) -> HTMLResponse:
    state = _state(request)
    service_date = _parse_date(date_)

    modes = ["train", "bus", "ferry", "local_transit"]
    if include_freerider:
        modes.append("freerider")
    if include_flights:
        modes.append("flight")

    constraints = _constraints(
        max_hours, max_transfers, include_freerider, modes, earliest, service_date
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
