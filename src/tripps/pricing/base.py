"""Shared machinery for price adapters: money conversion, politeness, call budgets.

Every adapter in this package talks to an endpoint that is either undocumented or not
meant for us. Two rules follow, and they are enforced here rather than trusted to each
adapter:

1. Never hammer. A per-source minimum interval and a per-search call budget mean one user
   query cannot fan out into a hundred requests against `global.api.flixbus.com`.
2. Never raise. A dead upstream must degrade to `Quote.unavailable(...)` with a deeplink,
   because an exception here empties the user's result list for a leg we could still
   describe perfectly well.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from ..config import PricingBudget
from ..interfaces import HealthState, PriceAdapter, SourceHealth
from ..models import Quote

log = logging.getLogger(__name__)


def ore_from_major(amount: float | int | str) -> int:
    """Convert a major-currency amount (439, "43.99") to integer öre.

    Rounds half-up at the öre. Prices arrive from JSON as floats, and 43.99 * 100 is
    4398.999... in binary floating point; truncating would lose an öre on every such fare.
    """
    value = float(amount)
    if value < 0:
        raise ValueError(f"negative amount: {amount!r}")
    return int(value * 100 + 0.5)


class RateLimiter:
    """Serializes calls to one host and spaces them by at least `min_interval`."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, min_interval)
        self._lock = asyncio.Lock()
        self._last: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            elapsed = loop.time() - self._last
            if self._last and elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last = loop.time()


class BudgetExceeded(RuntimeError):
    """Raised inside the orchestrator, never surfaced to the user."""


@dataclass(slots=True)
class CallBudget:
    """Per-search, per-source call ceiling.

    Without this, pricing `max_candidates_to_price` itineraries with several legs each,
    across several sources, is an easy hundred requests for one search box submission.
    """

    limit: int
    used: dict[str, int] = field(default_factory=dict)

    def consume(self, source: str) -> None:
        count = self.used.get(source, 0)
        if count >= self.limit:
            raise BudgetExceeded(f"{source}: exhausted {self.limit} calls for this search")
        self.used[source] = count + 1

    def remaining(self, source: str) -> int:
        return max(0, self.limit - self.used.get(source, 0))

    def spent(self) -> int:
        return sum(self.used.values())

    @classmethod
    def from_settings(cls, budget: PricingBudget) -> CallBudget:
        return cls(limit=budget.max_calls_per_source_per_search)


class HttpPriceAdapter(PriceAdapter):
    """Base for adapters that speak HTTP. Owns a client, a rate limiter, and a health flag."""

    name: str = "http"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "tripps/0.1",
        timeout: float = 20.0,
        min_interval: float = 0.35,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._user_agent = user_agent
        self._timeout = timeout
        self._limiter = RateLimiter(min_interval)
        self._health = SourceHealth(self.name, HealthState.OK)
        self._budget: CallBudget | None = None

    def set_budget(self, budget: CallBudget | None) -> None:
        """Attach this search's budget.

        The budget is charged in `get_json`, at the moment a request actually leaves the
        process, rather than once per leg the orchestrator wants priced. Those are very
        different numbers: one FlixBus response carries every departure of the day for a
        city pair, so pricing six of them is one request. Charging per leg would refuse to
        price journeys that cost nothing extra to price.
        """
        self._budget = budget

    async def http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                follow_redirects=True,
            )
        return self._client

    async def get_json(
        self, url: str, params: dict | None = None, *, budget_key: str | None = None
    ) -> object:
        """Fetch JSON, charging the search budget for the request.

        `budget_key` lets an adapter separate its expensive calls from its cheap ones. A
        FlixBus price search and a city-name autocomplete both cost one HTTP request, but
        the autocomplete is a tiny metadata lookup whose answer never changes, and letting
        it drain the same allowance starves the searches that actually produce prices.
        """
        if self._budget is not None:
            self._budget.consume(budget_key or self.name)  # raises BudgetExceeded
        await self._limiter.acquire()
        client = await self.http()
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def mark_ok(self) -> None:
        self._health = SourceHealth(self.name, HealthState.OK)

    def mark_degraded(self, detail: str) -> None:
        log.warning("%s degraded: %s", self.name, detail)
        self._health = SourceHealth(self.name, HealthState.DEGRADED, detail)

    def mark_down(self, detail: str) -> None:
        log.warning("%s down: %s", self.name, detail)
        self._health = SourceHealth(self.name, HealthState.DOWN, detail)

    async def health(self) -> SourceHealth:
        return self._health

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def unavailable(source: str, deeplink: str | None, reason: str) -> Quote:
    return Quote.unavailable(source=source, deeplink=deeplink, note=reason)


def exact(
    source: str,
    amount_ore: int,
    *,
    deeplink: str | None = None,
    fare_class: str | None = None,
    note: str | None = None,
) -> Quote:
    from ..models import PriceConfidence

    return Quote(
        source=source,
        amount_ore=amount_ore,
        confidence=PriceConfidence.EXACT,
        fare_class=fare_class,
        deeplink=deeplink,
        note=note,
        fetched_at=datetime.now(UTC),
    )


def estimated(
    source: str,
    amount_ore: int,
    *,
    deeplink: str | None = None,
    note: str | None = None,
) -> Quote:
    from ..models import PriceConfidence

    return Quote(
        source=source,
        amount_ore=amount_ore,
        confidence=PriceConfidence.ESTIMATED,
        deeplink=deeplink,
        note=note,
        fetched_at=datetime.now(UTC),
    )
