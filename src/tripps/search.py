"""The planner: resolve places, build the day's network, route, price, rank.

This is where the two phases meet. Phase 1 routes on schedules with price lower bounds;
phase 2 prices the survivors for real. Freerider cars and (later) flights are overlaid onto
the daily GTFS timetable per search, because their inventory is live and query-scoped.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .calibration import DISCOUNT_FLOOR_SCALE, load_calibrated_floors
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
from .passes import PassAdapter, TickitalAdapter, TickitalRental, honored_operators
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
    #: Hide itineraries with an unpriced leg (unless that would leave no results at all).
    require_priced: bool = True


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
        return resolve_stops(self.timetable, query, limit)

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

        pairs = [
            (source, target)
            for source in from_airports
            for target in to_airports
            if source.iata != target.iata
        ]
        # Each pair is an independent page scrape of several seconds. Run them concurrently:
        # done sequentially, four airport pairs turn a search into a 15-second wait before
        # routing even begins.
        results = await asyncio.gather(
            *(self._fetch_one_pair(s, t, service_date) for s, t in pairs)
        )
        collected: list[FlightOffer] = []
        for offers, pair_notes in results:
            collected.extend(offers)
            notes.extend(pair_notes)
        return collected, notes

    async def _fetch_one_pair(
        self, source, target, service_date: date
    ) -> tuple[list[FlightOffer], list[str]]:
        notes: list[str] = []
        key = cache_key(source, target, service_date)
        if self.db is not None:
            hit = self.db.cache_get(key)
            if hit is not None:
                payload, _stale = hit
                return [FlightOffer.from_dict(x) for x in payload], notes
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
                    notes.append(
                        f"Using cached {source.iata}-{target.iata} flight prices, "
                        "which may be out of date."
                    )
                    return [FlightOffer.from_dict(x) for x in payload], notes
            return [], notes
        if self.db is not None:
            self.db.cache_put(key, [o.to_dict() for o in offers], FLIGHT_CACHE_TTL_SECONDS)
        return offers, notes

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
        held_cards: list[str] | None = None,
        tickital_rentals: list[TickitalRental] | None = None,
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
            elif isinstance(adapter, PassAdapter):
                # Owned cards only - they zero covered legs before any paid source.
                adapter.prepare(network, held_cards or [])
            elif isinstance(adapter, TickitalAdapter):
                # Rentals are a paid fallback + itinerary-level coupon, priced after the real
                # sources; bind them here over this timetable.
                adapter.prepare(network, tickital_rentals or [])

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

        # Pass-aware routing: a held card can make a leg free, but the price floor is derived
        # from distance and would prune a covered-cheap itinerary before it is priced. Zeroing
        # the floor for every operator a held card honors keeps those journeys on the frontier;
        # zero is a valid lower bound, so the floor invariant still holds. A tickital rental
        # frees the same operators, but only on dates inside its window, so it contributes only
        # when this search date is in-window.
        # A discounted traveller pays under every adult fare the floors were fitted to, so the
        # adult model would bound their journeys ABOVE the truth and prune the cheapest one.
        # With a database, use the floors calibrated for this category (falling back inside
        # `load_calibrated_floors` to a scaled adult fit); without one, scale what we hold.
        floors = self.floors
        if not constraints.passenger.is_adult:
            floors = (
                load_calibrated_floors(self.db, constraints.passenger)
                if self.db is not None
                else floors.scaled(DISCOUNT_FLOOR_SCALE)
            )
        pass_ids = list(held_cards or [])
        pass_ids += [r.provider_id for r in (tickital_rentals or []) if r.active_on(service_date)]
        card_ops = honored_operators(pass_ids) if pass_ids else frozenset()
        if card_ops:
            floors = floors.with_operators_zeroed(card_ops)

        result = run_mcraptor(network, floors, query)
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
            candidates,
            constraints,
            max_results=self.options.max_results,
            require_priced=self.options.require_priced,
            # Operators whose floor was zeroed above: a paid quote below their calibrated floor
            # pruned nothing (the router used 0), so it must not read as a floor violation.
            zeroed_operators=card_ops,
            # The very model the router just used, so the violation detector and the
            # calibration samples describe the bound that actually shaped this search.
            floors=floors,
        )
        stats.priced = len(priced.itineraries)
        stats.upstream_calls = priced.calls_made

        warnings = list(priced.warnings) + stats.notes
        if not candidates:
            warnings.append("No journey found on this date.")
        warnings.extend(
            self._freerider_suggestions(origin, destination, service_date, offers or [])
        )

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


    def _freerider_suggestions(
        self, origin: Stop, destination: Stop, service_date: date, offers: list[FreeriderOffer]
    ) -> list[str]:
        """Surface free cars on this route available on *other* nearby dates.

        A date-specific search hides the killer feature: a car available three days later than
        the one you searched. This peeks at the full inventory and mentions the soonest one or
        two on the route that are not already on the searched date.
        """
        from .watcher import upcoming_cars_on_route

        if not offers:
            return []
        on_route = upcoming_cars_on_route(
            offers, origin.lat, origin.lon, destination.lat, destination.lon
        )
        # Skip cars whose window already covers the searched date - those are in the results.
        other_dates = [
            o for o in on_route if not (o.available_at.date() <= service_date <= o.latest_return.date())
        ]
        notes = []
        for offer in other_dates[:2]:
            notes.append(
                f"Free Hertz car on this route: {offer.pickup.name} → {offer.dropoff.name}, "
                f"available {offer.available_at:%b %d}. Book on hertzfreerider.se."
            )
        return notes


#: A factory that hands back a Planner bound to a given service date. The app's is backed by
#: its per-date timetable cache; the CLI builds one per call. Kept abstract so round-trip and
#: window search can drive many dates without knowing how timetables are loaded.
PlannerFactory = Callable[[date], "Planner"]


def _cheapest_priced(response: SearchResponse) -> Itinerary | None:
    priced = [i for i in response.itineraries if i.total_price_ore is not None]
    return min(priced, key=lambda i: i.total_price_ore) if priced else None


@dataclass(slots=True)
class RoundTrip:
    outbound: SearchResponse
    inbound: SearchResponse
    #: The rentals in force for this round trip, needed to dedup a one-time period cost that
    #: each direction's coupon charges independently. Populated by `round_trip()`.
    tickital_rentals: list[TickitalRental] = field(default_factory=list)

    @property
    def cheapest_outbound(self) -> Itinerary | None:
        return _cheapest_priced(self.outbound)

    @property
    def cheapest_inbound(self) -> Itinerary | None:
        return _cheapest_priced(self.inbound)

    def _price_of(self, rental_id: int) -> int | None:
        return next((r.price_ore for r in self.tickital_rentals if r.id == rental_id), None)

    def _charged_rentals(self, itin: Itinerary | None) -> set[int]:
        """Rental ids whose one-time period cost is actually charged in this itinerary.

        The coupon marks every rewritten leg with `coupon_rental_id`, but only the single
        charged leg carries the full `price_ore` (its siblings are zeroed), so matching on the
        amount picks the charge exactly once and by identity - two rentals with an equal price
        stay distinct.
        """
        if itin is None:
            return set()
        charged: set[int] = set()
        for leg in itin.legs:
            q = leg.quote
            if q is None or q.coupon_rental_id is None:
                continue
            price = self._price_of(q.coupon_rental_id)
            if price is not None and q.amount_ore == price:
                charged.add(q.coupon_rental_id)
        return charged

    @property
    def shared_rentals(self) -> set[int]:
        """Rentals charged in BOTH directions; their period cost is one-time, not per-direction."""
        return self._charged_rentals(self.cheapest_outbound) & self._charged_rentals(
            self.cheapest_inbound
        )

    @property
    def total_price_ore(self) -> int | None:
        out, back = self.cheapest_outbound, self.cheapest_inbound
        if out is None or back is None or out.total_price_ore is None or back.total_price_ore is None:
            return None
        total = out.total_price_ore + back.total_price_ore
        for rental_id in self.shared_rentals:
            price = self._price_of(rental_id)
            if price is not None:
                total -= price  # count the one-time period cost once, not once per direction
        return total

    @property
    def warnings(self) -> list[str]:
        notes: list[str] = []
        for rental_id in sorted(self.shared_rentals):
            price = self._price_of(rental_id)
            if price is not None:
                notes.append(
                    f"A tickital rental's one-time period cost of {price / 100:.0f} SEK is "
                    "counted once across the outbound and inbound totals, not on both."
                )
        return notes


async def round_trip(
    make_planner: PlannerFactory,
    origin: str,
    destination: str,
    outbound_date: date,
    return_date: date,
    *,
    constraints: SearchConstraints | None = None,
    offers: list[FreeriderOffer] | None = None,
    held_cards: list[str] | None = None,
    tickital_rentals: list[TickitalRental] | None = None,
) -> RoundTrip:
    """Search there and back, each on its own date's timetable, and combine the totals.

    The two legs are genuinely separate searches - a return journey runs a different calendar
    and prices independently - so this drives the planner twice and pairs the cheapest of each.
    """
    out_planner = make_planner(outbound_date)
    out_resp, _ = await out_planner.search(
        origin, destination, outbound_date, constraints=constraints, offers=offers,
        held_cards=held_cards, tickital_rentals=tickital_rentals,
    )
    in_planner = make_planner(return_date)
    in_resp, _ = await in_planner.search(
        destination, origin, return_date, constraints=constraints, offers=offers,
        held_cards=held_cards, tickital_rentals=tickital_rentals,
    )
    return RoundTrip(
        outbound=out_resp, inbound=in_resp, tickital_rentals=list(tickital_rentals or [])
    )


@dataclass(slots=True)
class DayFare:
    date: date
    cheapest: Itinerary | None

    @property
    def price_ore(self) -> int | None:
        return self.cheapest.total_price_ore if self.cheapest is not None else None

    @property
    def price_sek(self) -> float | None:
        p = self.price_ore
        return None if p is None else p / 100


async def cheapest_over_window(
    make_planner: PlannerFactory,
    origin: str,
    destination: str,
    start_date: date,
    days: int,
    *,
    constraints: SearchConstraints | None = None,
    offers: list[FreeriderOffer] | None = None,
    held_cards: list[str] | None = None,
    tickital_rentals: list[TickitalRental] | None = None,
) -> list[DayFare]:
    """The cheapest priced itinerary for each day in a window - a fare calendar.

    Swedish long-distance fares are yield-managed (one Stockholm->Goteborg day ranged 445-805
    SEK on SJ alone), so *when* to travel often saves more than *how*. Each day is a full
    search; the per-date timetable cache makes the second and later days cheap. Returned in
    date order; the caller sorts by price to find the bargain day.
    """
    results: list[DayFare] = []
    for offset in range(days):
        day = start_date + timedelta(days=offset)
        planner = make_planner(day)
        response, _ = await planner.search(
            origin, destination, day, constraints=constraints, offers=offers,
            held_cards=held_cards, tickital_rentals=tickital_rentals,
        )
        results.append(DayFare(date=day, cheapest=_cheapest_priced(response)))
    return results


def window_rental_notes(fares: list[DayFare], rentals: list[TickitalRental] | None) -> list[str]:
    """Notes flagging a tickital rental charged on 2+ days of a fare calendar.

    Each in-window day is an independent search that charges the full one-time period cost, so
    the same figure appearing on several days would misread as a per-day expense. This says the
    cost is shared across the window: choosing an extra in-window day costs about 0 more.
    """
    if not rentals:
        return []
    price_of = {r.id: r.price_ore for r in rentals if r.id is not None}
    days_charged: dict[int, int] = {}
    for fare in fares:
        if fare.cheapest is None:
            continue
        for rental_id in _charged_rental_ids(fare.cheapest, price_of):
            days_charged[rental_id] = days_charged.get(rental_id, 0) + 1
    notes: list[str] = []
    for rental_id, count in sorted(days_charged.items()):
        if count >= 2:
            notes.append(
                f"A tickital rental's one-time period cost of {price_of[rental_id] / 100:.0f} "
                f"SEK is shown on {count} days, but it is paid once for the whole window - an "
                "extra in-window day costs about 0 more, not the full figure."
            )
    return notes


def _charged_rental_ids(itin: Itinerary, price_of: dict[int, int]) -> set[int]:
    """Rental ids charged (the full-price leg, not a zeroed sibling) in one itinerary."""
    charged: set[int] = set()
    for leg in itin.legs:
        q = leg.quote
        if q is None or q.coupon_rental_id is None:
            continue
        price = price_of.get(q.coupon_rental_id)
        if price is not None and q.amount_ore == price:
            charged.add(q.coupon_rental_id)
    return charged


def _fold(text: str) -> str:
    """Casefold and strip diacritics, so "malmo"/"goteborg" match "Malmö"/"Göteborg" for a
    traveller without the å/ä/ö keys. Swedish speakers who type the letters still match too."""
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", text.strip().casefold())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def resolve_stops(timetable: Timetable, query: str, limit: int = 8) -> list[Stop]:
    """Match a typed place name against timetable stops.

    Exact match first, then prefix, then contains; within each class, the busiest stop wins.
    Busyness matters: in the national feed "Stockholm" prefix-matches a dozen ferry landings
    before it matches Stockholm Centralstation, and a traveller typing "Stockholm" means the
    station. Route count is a decent stand-in for importance and costs nothing, since the
    stop-to-route index already exists for the router.

    A free function as well as a Planner method, so tools that hold only a timetable (the
    Freerider watch CLI) can resolve a place without building a whole planner.
    """
    needle = _fold(query)
    if not needle:
        return []

    buckets: list[list[tuple[int, Stop]]] = [[], [], []]
    for index, stop in enumerate(timetable.stops):
        name = _fold(stop.name)
        if name == needle:
            rank = 0
        elif name.startswith(needle):
            rank = 1
        elif needle in name:
            rank = 2
        else:
            continue
        buckets[rank].append((len(timetable.stop_routes[index]), stop))

    ordered: list[Stop] = []
    for bucket in buckets:
        bucket.sort(key=lambda pair: (-pair[0], pair[1].name))
        ordered.extend(stop for _routes, stop in bucket)
    return ordered[:limit]


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
