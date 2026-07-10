"""Command line entry points: fetch the feed, poll cars, search, serve."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date

import httpx

from .config import get_settings
from .db import Database
from .ingest.flights import GoogleFlightsProvider, NullFlightProvider
from .ingest.freerider import (
    FreeriderClient,
    FreeriderCostModel,
    offers_to_log_rows,
    only_within,
    parse_offers,
    schema_drift,
)
from .ingest.gtfs import GtfsConfig, load_timetable
from .models import SearchConstraints, TransportMode
from .pricing.flights import FlightAdapter
from .pricing.flixbus import FlixBusAdapter
from .pricing.freerider import FreeriderAdapter
from .pricing.operators import DeeplinkAdapter, StaticFareAdapter
from .pricing.orchestrator import PricingOrchestrator
from .pricing.sj import SJAdapter
from .routing.floors import PriceFloorModel
from .search import Planner, SearchOptions, summarize
from .timeutil import now_local


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


async def _fetch_gtfs() -> int:
    """Download the national GTFS feed (CC0, refreshed daily).

    Trafiklab documents an API key on this download, but the endpoint currently serves the
    feed unauthenticated. The key is appended when configured, so the command keeps working
    if that changes; it is not required today.
    """
    settings = get_settings()
    settings.ensure_dirs()

    url = settings.gtfs_url
    if settings.trafiklab_gtfs_key:
        url = f"{url}?key={settings.trafiklab_gtfs_key}"
    target = settings.gtfs_zip_path
    print(f"downloading GTFS feed -> {target}")
    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                print(f"error: feed returned HTTP {response.status_code}", file=sys.stderr)
                return 1
            written = 0
            with target.open("wb") as handle:
                async for chunk in response.aiter_bytes(1 << 20):
                    handle.write(chunk)
                    written += len(chunk)
    print(f"wrote {written / 1e6:.1f} MB")
    return 0


async def _freerider(as_json: bool) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    client = FreeriderClient(base_url=settings.freerider_base, user_agent=settings.user_agent)
    cost = FreeriderCostModel()
    try:
        raw = await client.fetch_raw("SWEDEN")
        offers = only_within(parse_offers(raw), "se")
    finally:
        await client.aclose()

    drift = schema_drift(offers)
    if drift:
        print(f"WARNING: Freerider schema drift, cost model may be wrong: {drift[0]}", file=sys.stderr)

    db = Database(settings.db_path)
    db.log_freerider_offers(offers_to_log_rows(offers, raw))
    db.close()

    if as_json:
        import json

        print(json.dumps([o.pickup.city + "->" + o.dropoff.city for o in offers], ensure_ascii=False))
        return 0

    print(f"{len(offers)} Freerider cars within Sweden\n")
    for offer in sorted(offers, key=lambda o: o.direct_km):
        estimate = cost.estimate_ore(offer) / 100
        price = "free" if estimate == 0 else f"~{estimate:.0f} SEK fuel"
        print(
            f"  {offer.pickup.city[:18]:20} -> {offer.dropoff.city[:18]:20} "
            f"{offer.direct_km:6.0f} km  {offer.direct_minutes // 60}h{offer.direct_minutes % 60:02d}m  "
            f"{price:16} until {offer.latest_return:%m-%d %H:%M}  {offer.car_model[:24]}"
        )
    return 0


async def _search(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    service_date = date.fromisoformat(args.date) if args.date else now_local().date()

    if not settings.gtfs_zip_path.exists():
        print(f"error: no GTFS feed at {settings.gtfs_zip_path}; run `tripps fetch-gtfs`", file=sys.stderr)
        return 2

    print(f"loading timetable for {service_date}...", file=sys.stderr)
    timetable, stats = load_timetable(settings.gtfs_zip_path, service_date, GtfsConfig())
    print(
        f"  {stats.stops} stops, {stats.routes} routes, {stats.trips} trips, "
        f"{len(stats.agencies)} operators",
        file=sys.stderr,
    )

    offers = []
    if not args.no_freerider:
        client = FreeriderClient(base_url=settings.freerider_base, user_agent=settings.user_agent)
        try:
            offers = only_within(parse_offers(await client.fetch_raw("SWEDEN")), "se")
            print(f"  {len(offers)} Freerider cars", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"  freerider unavailable: {exc}", file=sys.stderr)
        finally:
            await client.aclose()

    db = Database(settings.db_path)
    adapters = [
        FlixBusAdapter(
            base_url=settings.flixbus_base,
            user_agent=settings.user_agent,
            min_interval=settings.budget.min_interval_seconds,
            db=db,
        ),
        SJAdapter(
            user_agent=settings.user_agent,
        ),
        FreeriderAdapter(cost_model=FreeriderCostModel()),
        FlightAdapter(),
        StaticFareAdapter.from_file(settings.data_dir / "fares.json")
        if (settings.data_dir / "fares.json").exists()
        else StaticFareAdapter(),
        DeeplinkAdapter(),
    ]
    orchestrator = PricingOrchestrator(
        adapters, db, budget=settings.budget, ttl=settings.ttl, floors=PriceFloorModel()
    )

    allowed = {TransportMode.TRAIN, TransportMode.BUS, TransportMode.FERRY, TransportMode.WALK}
    if not args.no_freerider:
        allowed.add(TransportMode.FREERIDER)
    if args.flights:
        allowed.add(TransportMode.FLIGHT)

    planner = Planner(
        timetable,
        orchestrator,
        db=db,
        options=SearchOptions(max_results=args.limit, include_flights=args.flights),
        flight_provider=GoogleFlightsProvider() if args.flights else NullFlightProvider(),
    )

    try:
        response, run_stats = await planner.search(
            args.origin,
            args.destination,
            service_date,
            constraints=SearchConstraints(
                allowed_modes=frozenset(allowed),
                max_transfers=args.max_transfers,
                max_duration_seconds=int(args.max_hours * 3600) if args.max_hours else None,
                include_freerider=not args.no_freerider,
            ),
            offers=offers,
        )
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        for adapter in adapters:
            await adapter.aclose()

    print(f"\n{response.origin.name} -> {response.destination.name} on {response.date}\n")
    if not response.itineraries:
        print("  no itinerary found")
    for index, itin in enumerate(response.itineraries, 1):
        marker = "*" if index == 1 else " "
        print(f" {marker} {index}. {summarize(itin)}")
        for leg in itin.legs:
            if leg.mode is TransportMode.WALK:
                continue
            quote = leg.quote
            if quote is None or quote.amount_ore is None:
                cost = "no price"
            elif quote.amount_ore == 0:
                cost = "free"
            else:
                cost = f"{quote.amount_sek:.0f} SEK"
            print(
                f"       {leg.departure:%H:%M}-{leg.arrival:%H:%M} {leg.mode.value:12} "
                f"{leg.from_stop.name[:24]:26} -> {leg.to_stop.name[:24]:26} {cost}"
            )
        print()

    if response.warnings:
        print("warnings:")
        for warning in dict.fromkeys(response.warnings):
            print(f"  - {warning}")

    print(
        f"\n{run_stats.candidates} candidates, {run_stats.upstream_calls} upstream calls, "
        f"{run_stats.freerider_offers} cars, {run_stats.flight_offers} flights",
        file=sys.stderr,
    )
    db.close()
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("tripps.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tripps", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch-gtfs", help="download the national GTFS feed")

    fr = sub.add_parser("freerider", help="list free Hertz Freerider cars in Sweden")
    fr.add_argument("--json", action="store_true", dest="as_json")

    se = sub.add_parser("search", help="find the cheapest way from A to B")
    se.add_argument("origin")
    se.add_argument("destination")
    se.add_argument("--date", help="YYYY-MM-DD, defaults to today")
    se.add_argument("--limit", type=int, default=5)
    se.add_argument("--max-transfers", type=int, default=None)
    se.add_argument("--max-hours", type=float, default=None)
    se.add_argument("--no-freerider", action="store_true")
    se.add_argument("--flights", action="store_true", help="also scrape domestic flight prices")

    sv = sub.add_parser("serve", help="run the web UI and JSON API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "fetch-gtfs":
        return asyncio.run(_fetch_gtfs())
    if args.command == "freerider":
        return asyncio.run(_freerider(args.as_json))
    if args.command == "search":
        return asyncio.run(_search(args))
    if args.command == "serve":
        return _serve(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
