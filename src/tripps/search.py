"""The planner: resolve places, build the day's network, route, price, rank.

This is where the two phases meet. Phase 1 routes on schedules with price lower bounds;
phase 2 prices the survivors for real. Freerider cars and (later) flights are overlaid onto
the daily GTFS timetable per search, because their inventory is live and query-scoped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from .db import Database
from .ingest.airports import nearest_airports, resolve_airport_stop
from .ingest.flights import (
    FLIGHT_CACHE_TTL_SECONDS,
    FlightOffer,
    FlightProvider,
    NullFlightProvider,
    cache_key,
    flight_route_addition,
)
from .ingest.freerider import FreeriderCostModel, FreeriderOffer
from .models import (
    Itinerary,
    SearchConstraints,
    SearchResponse,
    Stop,
    TransportMode,
)
from .pricing.flights import FlightAdapter
from .pricing.freerider import FreeriderAdapter
from .pricing.orchestrator import PricingOrchestrator
from .routing.floors import PriceFloorModel
from .routing.journey import dedupe_itineraries, label_to_itinerary
from .routing.mcraptor import RaptorQuery, run_mcraptor
from .routing.synthetic import connect_freerider_stations, freerider_additions
from .routing.timetable import (
    INFINITY,
    RouteAddition,
    Timetable,
    haversine_km,
    overlay_timetable,
)
from .timeutil import now_local, to_service_seconds

log = logging.getLogger(__name__)

#: Radius within which a typed place name also covers the neighbouring stops that serve the
#: same place. Stockholm Centralstation and Stockholm Cityterminalen are 300 m apart and are
#: for practical purposes one origin; Landvetter Airport is 25 km from Goteborg C and is
#: emphatically not the same destination. Airports stay reachable because the feed's own
#: airport trains and coaches call at them as ordinary routes.
STOP_MATCH_RADIUS_KM = 2.0


@dataclass(slots=True)
class SearchOptions:
    max_rounds: int = 6
    min_transfer_seconds: int = 300
    max_results: int = 5
    include_freerider: bool = True
    freerider_step_minutes: int = 30
    include_flights: bool = True
    #: How many airports near each endpoint to search. Each one is a page scrape.
    airports_per_endpoint: int = 2


@dataclass(slots=True)
class PlannerStats:
    candidates: int = 0
    priced: int = 0
    upstream_calls: int = 0
    freerider_offers: int = 0
    flight_offers: int = 0
    rounds: int = 0
    notes: list[str] = field(default_factory=list)


class Planner:
    """Owns the daily timetable and answers search queries against it."""

    def __init__(
        self,
        timetable: Timetable,
        orchestrator: PricingOrchestrator,
        *,
        floors: PriceFloorModel | None = None,
        cost_model: FreeriderCostModel | None = None,
        db: Database | None = None,
        options: SearchOptions | None = None,
        flight_provider: FlightProvider | None = None,
    ) -> None:
        self.timetable = timetable
        self.orchestrator = orchestrator
        self.floors = floors or PriceFloorModel()
        self.cost_model = cost_model or FreeriderCostModel()
        self.db = db
        self.options = options or SearchOptions()
        self.flight_provider: FlightProvider = flight_provider or NullFlightProvider()

    # --- place resolution -------------------------------------------------

    def resolve_stops(self, query: str, limit: int = 8) -> list[Stop]:
        """Match a typed place name against timetable stops.

        Exact match first, then prefix, then contains; within each class, the busiest stop
        wins. Busyness matters: in the national feed "Stockholm" prefix-matches a dozen
        ferry landings before it matches Stockholm Centralstation, and a traveller typing
        "Stockholm" means the station. Route count is a decent stand-in for importance and
        costs nothing, since the stop-to-route index already exists for the router.
        """
        needle = query.strip().casefold()
        if not needle:
            return []

        buckets: list[list[tuple[int, Stop]]] = [[], [], []]
        for index, stop in enumerate(self.timetable.stops):
            name = stop.name.casefold()
            if name == needle:
                rank = 0
            elif name.startswith(needle):
                rank = 1
            elif needle in name:
                rank = 2
            else:
                continue
            buckets[rank].append((len(self.timetable.stop_routes[index]), stop))

        ordered: list[Stop] = []
        for bucket in buckets:
            bucket.sort(key=lambda pair: (-pair[0], pair[1].name))
            ordered.extend(stop for _routes, stop in bucket)
        return ordered[:limit]

    def _stop_group(self, anchor: Stop, radius_km: float = STOP_MATCH_RADIUS_KM) -> list[Stop]:
        """All stops that serve the same place, so a search is not tied to one platform.

        A traveller leaving "Stockholm" may depart from the central station or from
        Cityterminalen next door; a Freerider car may wait at an airport across town.
        """
        return [
            stop
            for stop in self.timetable.stops
            if haversine_km(anchor.lat, anchor.lon, stop.lat, stop.lon) <= radius_km
        ]

    def _nearby_stops(self, lat: float, lon: float, radius_km: float) -> list[tuple[Stop, float]]:
        found = []
        for stop in self.timetable.stops:
            km = haversine_km(lat, lon, stop.lat, stop.lon)
            if km <= radius_km:
                found.append((stop, km))
        found.sort(key=lambda pair: pair[1])
        return found[:12]

    # --- network assembly -------------------------------------------------

    async def fetch_flight_offers(
        self, origin: Stop, destination: Stop, service_date: date
    ) -> tuple[list[FlightOffer], list[str]]:
        """Priced direct domestic flights between airports serving these two places.

        Each airport pair is one page scrape, so results are cached for an hour and a dead
        provider degrades to "no flights", never to an exception. Stale offers are used
        rather than dropped when the provider is down, with a warning.
        """
        notes: list[str] = []
        from_airports = nearest_airports(
            origin.lat, origin.lon, limit=self.options.airports_per_endpoint
        )
        to_airports = nearest_airports(
            destination.lat, destination.lon, limit=self.options.airports_per_endpoint
        )

        collected: list[FlightOffer] = []
        for source in from_airports:
            for target in to_airports:
                if source.iata == target.iata:
                    continue
                key = cache_key(source, target, service_date)
                if self.db is not None:
                    hit = self.db.cache_get(key)
                    if hit is not None:
                        payload, _stale = hit
                        collected.extend(FlightOffer.from_dict(x) for x in payload)
                        continue
                try:
                    offers = await self.flight_provider.search(source, target, service_date)
                except Exception as exc:  # noqa: BLE001 - a scraper must not kill the search
                    log.warning("flight search %s-%s failed: %s", source.iata, target.iata, exc)
                    notes.append(
                        f"Flight prices for {source.iata}-{target.iata} are unavailable "
                        f"({type(exc).__name__})."
                    )
                    if self.db is not None:
                        stale = self.db.cache_get(key, allow_stale=True)
                        if stale is not None:
                            payload, _ = stale
                            collected.extend(FlightOffer.from_dict(x) for x in payload)
                            notes.append(
                                f"Using cached {source.iata}-{target.iata} flight prices, "
                                "which may be out of date."
                            )
                    continue
                if self.db is not None:
                    self.db.cache_put(
                        key, [o.to_dict() for o in offers], FLIGHT_CACHE_TTL_SECONDS
                    )
                collected.extend(offers)
        return collected, notes

    def _flight_additions(
        self, offers: list[FlightOffer], service_date: date
    ) -> tuple[list[RouteAddition], list[FlightOffer]]:
        """Group offers by airport pair and attach them to the feed's own airport stops.

        An airport with no timetabled stop nearby is unreachable on the ground, so a flight
        into it would produce an itinerary nobody can complete. Those offers are dropped.
        """
        registry = {a.iata: a for a in _airports_in(offers)}
        by_pair: dict[tuple[str, str], list[FlightOffer]] = {}
        for offer in offers:
            by_pair.setdefault((offer.from_iata, offer.to_iata), []).append(offer)

        additions: list[RouteAddition] = []
        used: list[FlightOffer] = []
        for (from_iata, to_iata), group in by_pair.items():
            origin_airport, target_airport = registry.get(from_iata), registry.get(to_iata)
            if origin_airport is None or target_airport is None:
                continue
            origin_stop = resolve_airport_stop(self.timetable, origin_airport)
            target_stop = resolve_airport_stop(self.timetable, target_airport)
            if origin_stop is None or target_stop is None:
                log.info("skipping %s-%s: no ground access in the feed", from_iata, to_iata)
                continue
            addition = flight_route_addition(group, service_date, origin_stop, target_stop)
            if addition is not None:
                additions.append(addition)
                used.extend(group)
        return additions, used

    def build_network(
        self,
        service_date: date,
        offers: list[FreeriderOffer],
        *,
        now: datetime | None = None,
        include_freerider: bool = True,
        flight_offers: list[FlightOffer] | None = None,
    ) -> tuple[Timetable, list[FreeriderOffer], list[FlightOffer]]:
        """Overlay live Freerider cars and flight offers onto the day's timetable."""
        additions: list[RouteAddition] = []
        edges: list[tuple[str, str, int]] = []
        used_cars: list[FreeriderOffer] = []
        used_flights: list[FlightOffer] = []

        if include_freerider and offers:
            now = now or now_local()
            car_additions = freerider_additions(
                offers,
                service_date,
                self.cost_model,
                now=now,
                step_minutes=self.options.freerider_step_minutes,
            )
            if car_additions:
                used_ids = {a.info.id.split(":", 1)[1] for a in car_additions}
                used_cars = [o for o in offers if str(o.route_id) in used_ids]
                additions.extend(car_additions)
                edges.extend(connect_freerider_stations(used_cars, self._nearby_stops))

        if flight_offers:
            flight_additions, used_flights = self._flight_additions(flight_offers, service_date)
            additions.extend(flight_additions)

        if not additions:
            return self.timetable, [], []
        return overlay_timetable(self.timetable, additions, edges), used_cars, used_flights

    # --- the search -------------------------------------------------------

    async def search(
        self,
        origin_query: str,
        destination_query: str,
        service_date: date,
        *,
        constraints: SearchConstraints | None = None,
        offers: list[FreeriderOffer] | None = None,
        now: datetime | None = None,
        departure_after: datetime | None = None,
    ) -> tuple[SearchResponse, PlannerStats]:
        constraints = constraints or SearchConstraints()
        stats = PlannerStats()

        origins = self.resolve_stops(origin_query)
        destinations = self.resolve_stops(destination_query)
        if not origins:
            raise LookupError(f"unknown place: {origin_query!r}")
        if not destinations:
            raise LookupError(f"unknown place: {destination_query!r}")
        origin, destination = origins[0], destinations[0]

        include_freerider = (
            self.options.include_freerider
            and constraints.include_freerider
            and TransportMode.FREERIDER in constraints.allowed_modes
        )
        include_flights = (
            self.options.include_flights and TransportMode.FLIGHT in constraints.allowed_modes
        )

        flight_offers: list[FlightOffer] = []
        if include_flights:
            flight_offers, flight_notes = await self.fetch_flight_offers(
                origin, destination, service_date
            )
            stats.notes.extend(flight_notes)

        network, used_offers, used_flights = self.build_network(
            service_date,
            offers or [],
            now=now,
            include_freerider=include_freerider,
            flight_offers=flight_offers,
        )
        stats.freerider_offers = len(used_offers)
        stats.flight_offers = len(used_flights)

        for adapter in self.orchestrator.adapters:
            if isinstance(adapter, FreeriderAdapter):
                adapter.load(used_offers)
            elif isinstance(adapter, FlightAdapter):
                adapter.load(used_flights)

        depart_at = departure_after or constraints.earliest_departure
        depart_seconds = (
            to_service_seconds(depart_at, service_date) if depart_at is not None else 0
        )
        latest_arrival = (
            to_service_seconds(constraints.latest_arrival, service_date)
            if constraints.latest_arrival is not None
            else INFINITY
        )

        origin_group = self._stop_group(origin)
        target_group = self._stop_group(destination)
        query = RaptorQuery(
            origins=[
                (network.index_of(stop.id), depart_seconds)
                for stop in origin_group
                if stop.id in network.stop_index
            ],
            targets={
                network.index_of(stop.id)
                for stop in target_group
                if stop.id in network.stop_index
            },
            max_rounds=(
                min(self.options.max_rounds, constraints.max_transfers + 1)
                if constraints.max_transfers is not None
                else self.options.max_rounds
            ),
            min_transfer_seconds=self.options.min_transfer_seconds,
            allowed_modes=constraints.allowed_modes,
            latest_arrival=latest_arrival,
        )
        if not query.origins or not query.targets:
            raise LookupError("origin and destination did not resolve onto the network")

        result = run_mcraptor(network, self.floors, query)
        stats.rounds = result.rounds_run
        if result.bag_capped:
            stats.notes.append("label bags were capped; results may be approximate")

        candidates = dedupe_itineraries(
            [label_to_itinerary(network, label, service_date) for label in result.labels]
        )
        stats.candidates = len(candidates)

        # Constraints that depend on wall-clock times are applied after reconstruction,
        # since the router only knows service-day seconds.
        if depart_at is not None:
            candidates = [c for c in candidates if c.departure >= depart_at]

        priced = await self.orchestrator.price(
            candidates, constraints, max_results=self.options.max_results
        )
        stats.priced = len(priced.itineraries)
        stats.upstream_calls = priced.calls_made

        warnings = list(priced.warnings) + stats.notes
        if not candidates:
            warnings.append("No journey found on this date.")

        return (
            SearchResponse(
                origin=origin,
                destination=destination,
                date=service_date.isoformat(),
                itineraries=priced.itineraries,
                warnings=warnings,
                source_status=priced.source_status,
            ),
            stats,
        )


def _airports_in(offers: list[FlightOffer]):
    """The airport objects referenced by a set of offers."""
    from .ingest.airports import load_airports

    registry = load_airports()
    codes = {o.from_iata for o in offers} | {o.to_iata for o in offers}
    return [registry[code] for code in codes if code in registry]


def summarize(itin: Itinerary) -> str:
    """One-line human summary, used by the CLI and by logs."""
    modes = "+".join(
        dict.fromkeys(leg.mode.value for leg in itin.legs if leg.mode is not TransportMode.WALK)
    )
    total = itin.total_price_sek
    price = f"{total:.0f} SEK" if total is not None else "price unavailable"
    hours, remainder = divmod(itin.duration_seconds, 3600)
    return (
        f"{itin.departure:%H:%M}-{itin.arrival:%H:%M} "
        f"({hours}h{remainder // 60:02d}m, {itin.transfers} transfers) "
        f"{modes}: {price} [{itin.price_confidence.value}]"
    )
