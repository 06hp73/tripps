"""Live canaries for the reverse-engineered price sources.

Every price source is an undocumented endpoint that will eventually change: SJ rotates its
subscription key, FlixBus / Tora / Freerider can reshape their JSON, the Tora WAF can
tighten. The contract tests in `tests/` guard against *known* shapes using recorded
fixtures, but nothing there talks to the live services. These canaries do.

Each probe drives the real endpoint through the same client the planner uses, then asserts
the one fact the adapter depends on - a price came back, the inventory parsed, the schema
still means what the cost model assumes. The result is a health state per source, so the
first sign a source broke is an operator reading `tripps canary`, not a user silently
getting no prices.

Probes never raise: an exception is a DOWN result with the reason. They are safe to run on
a schedule (cron) and cheap - one or two requests each.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

from .config import Settings, get_settings
from .ingest.freerider import (
    FreeriderClient,
    only_within,
    parse_offers,
    schema_drift,
)
from .interfaces import HealthState
from .pricing.flixbus import FlixBusAdapter
from .pricing.sj import SJAdapter
from .pricing.tora import ToraAdapter
from .timeutil import TZ


#: A weekday a few days out, so probed services are actually running (avoids "no departures
#: today because it is late" false alarms) yet close enough that fares exist.
def _probe_date(today: date | None = None) -> date:
    today = today or datetime.now(TZ).date()
    day = today + timedelta(days=3)
    # Nudge onto a weekday; weekend regional/commercial service is thinner.
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


@dataclass(frozen=True, slots=True)
class CanaryResult:
    name: str
    state: HealthState
    detail: str
    latency_ms: int

    @property
    def ok(self) -> bool:
        return self.state in (HealthState.OK, HealthState.DEGRADED)

    def line(self) -> str:
        mark = {"ok": "OK  ", "degraded": "WARN", "down": "DOWN", "disabled": "----"}[
            self.state.value
        ]
        return f"[{mark}] {self.name:16} {self.latency_ms:>6} ms  {self.detail}"


async def _run(name: str, coro) -> CanaryResult:
    """Wrap a probe body so it always yields a result, timing the call."""
    t = time.perf_counter()
    try:
        state, detail = await coro
    except Exception as exc:  # noqa: BLE001 - a probe reports failure, never raises
        state, detail = HealthState.DOWN, f"{type(exc).__name__}: {exc}"
    return CanaryResult(name, state, detail, int((time.perf_counter() - t) * 1000))


# --- individual probes -----------------------------------------------------


async def _probe_gtfs(settings: Settings) -> tuple[HealthState, str]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.head(settings.gtfs_url)
        if resp.status_code != 200:
            # Some CDNs reject HEAD; fall back to a ranged GET of the first bytes.
            resp = await client.get(settings.gtfs_url, headers={"Range": "bytes=0-1"})
        if resp.status_code not in (200, 206):
            return HealthState.DOWN, f"feed returned HTTP {resp.status_code}"
    return HealthState.OK, "national feed reachable"


async def _probe_flixbus(settings: Settings, day: date) -> tuple[HealthState, str]:
    adapter = FlixBusAdapter(base_url=settings.flixbus_base, min_interval=0.0)
    try:
        origin = await adapter.resolve_city("Stockholm")
        dest = await adapter.resolve_city("Göteborg")
        if not origin or not dest:
            return HealthState.DOWN, "city autocomplete returned no id"
        departures = await adapter.search(origin, dest, day)
        priced = [d for d in departures if d.bookable and d.total_with_fee_ore > 0]
        if not priced:
            return HealthState.DEGRADED, "search returned no bookable priced departures"
        cheapest = min(d.total_with_fee_ore for d in priced)
        return HealthState.OK, f"{len(priced)} departures, from {cheapest / 100:.0f} SEK"
    finally:
        await adapter.aclose()


async def _probe_sj(settings: Settings, day: date) -> tuple[HealthState, str]:
    adapter = SJAdapter(min_interval=0.0)
    try:
        key = await adapter.ensure_key()
        if not key:
            return HealthState.DOWN, "could not resolve subscription key"
        plid, departures = await adapter._day_departures("740000001", "740000002", day)
        direct = [d for d in departures if d.is_direct]
        if not direct:
            return HealthState.DEGRADED, "no direct Stockholm-Göteborg departures"
        price = await adapter._offer_price_ore(direct[0].departure_id, plid)
        if price is None:
            return HealthState.DEGRADED, "offers returned no purchasable fare"
        return HealthState.OK, f"{len(direct)} direct departures, from {price / 100:.0f} SEK"
    finally:
        await adapter.aclose()


async def _probe_tora(day: date) -> tuple[HealthState, str]:
    adapter = ToraAdapter(min_interval=0.0)
    try:
        near = datetime.combine(day, datetime.min.time().replace(hour=8), tzinfo=TZ)
        journeys = await adapter._day_offers("740000003", "740000002", day, near)
        priced = [j for j in journeys if j.sole_carrier and j.cheapest_ore]
        if not priced:
            return HealthState.DEGRADED, "offers returned no priced single-carrier journey"
        carriers = {j.sole_carrier for j in priced}
        cheapest = min(j.cheapest_ore for j in priced)
        return HealthState.OK, f"{len(priced)} journeys ({', '.join(sorted(carriers))[:40]}), from {cheapest / 100:.0f} SEK"
    finally:
        await adapter.aclose()


async def _probe_freerider(settings: Settings) -> tuple[HealthState, str]:
    client = FreeriderClient(base_url=settings.freerider_base, user_agent=settings.user_agent)
    try:
        raw = await client.fetch_raw("SWEDEN")
        offers = only_within(parse_offers(raw), "se")
        if not offers:
            return HealthState.DEGRADED, "inventory parsed but held no Swedish offers"
        drift = schema_drift(offers)
        if drift:
            # The distance-is-mileage-allowance assumption underpins the cost model.
            return HealthState.DEGRADED, f"schema drift: {drift[0]}"
        return HealthState.OK, f"{len(offers)} Swedish cars, schema intact"
    finally:
        await client.aclose()


# --- runner ----------------------------------------------------------------


async def run_canaries(
    settings: Settings | None = None, *, day: date | None = None
) -> list[CanaryResult]:
    """Probe every price source concurrently and return their health."""
    settings = settings or get_settings()
    day = day or _probe_date()
    results = await asyncio.gather(
        _run("gtfs-feed", _probe_gtfs(settings)),
        _run("flixbus", _probe_flixbus(settings, day)),
        _run("sj", _probe_sj(settings, day)),
        _run("tora", _probe_tora(day)),
        _run("freerider", _probe_freerider(settings)),
    )
    return list(results)


def persist_canaries(db, results: list[CanaryResult]) -> None:
    """Record the latest probe outcomes so /health and the UI can show them."""
    for r in results:
        db.set_health(f"canary:{r.name}", r.state.value, f"{r.detail} ({r.latency_ms}ms)")
