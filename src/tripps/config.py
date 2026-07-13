"""Runtime configuration.

Every external dependency is optional at import time. The planner must start, and
degrade honestly, when a key is missing or a host is down; only the features that
truly need a credential get disabled.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import TransportMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


class PricingBudget(BaseSettings):
    """The freshness/fan-out contract for phase-2 pricing.

    Without a hard cap, one user search fans out to (candidates x legs) upstream
    calls against undocumented endpoints. That is how a hobby project gets its IP
    blocked. Every adapter is called through this budget.
    """

    #: Max HTTP requests one search may send to a single source. Charged in the adapter at
    #: the moment a request leaves the process, so legs answered from an adapter's own memo
    #: (one FlixBus response covers every departure that day) cost nothing.
    max_calls_per_source_per_search: int = 12
    #: Max itineraries priced. Bounded by CPU and by the pricing deadline rather than by the
    #: network, because pricing many departures of one service reuses a single response.
    max_candidates_to_price: int = 30
    #: How many departures of the same journey shape get priced. The range query finds one
    #: per departure; sampling several is how the cheap late-night coach gets discovered,
    #: while a cap stops sixteen pickup times of one free car eating the budget.
    max_departures_per_pattern: int = 6
    #: Wall-clock ceiling for the whole phase-2 fan-out. Generous, because a cut-off here
    #: now *drops* a route (it fails the fully-priced filter) rather than merely showing it
    #: with a gap, so the ceiling must sit above a normal multi-operator pricing run.
    phase2_timeout_seconds: float = 35.0
    #: Per-source politeness delay between calls, seconds.
    min_interval_seconds: float = 0.35

    model_config = SettingsConfigDict(env_prefix="TRIPPS_BUDGET_")


class CacheTTL(BaseSettings):
    """Per-source quote cache lifetimes, seconds.

    Rail is shortest because it is yield-managed and re-prices continuously.
    Freerider entries are additionally bounded by each offer's own expireTime.
    """

    rail: int = 600
    bus: int = 1800
    flight: int = 3600
    freerider: int = 300
    ferry: int = 3600
    local_transit: int = 86400

    model_config = SettingsConfigDict(env_prefix="TRIPPS_TTL_")

    def for_mode(self, mode: TransportMode) -> int:
        return {
            TransportMode.TRAIN: self.rail,
            TransportMode.BUS: self.bus,
            TransportMode.FLIGHT: self.flight,
            TransportMode.FREERIDER: self.freerider,
            TransportMode.FERRY: self.ferry,
            TransportMode.LOCAL_TRANSIT: self.local_transit,
            TransportMode.WALK: self.local_transit,
        }[mode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRIPPS_", env_file=".env", extra="ignore"
    )

    # --- Data sources -------------------------------------------------------
    #: Trafiklab documents an API key on the GTFS download, but the endpoint currently
    #: serves the national feed unauthenticated (verified 2026-07-10). The key is sent when
    #: present so the app keeps working if that changes.
    trafiklab_gtfs_key: str | None = None
    #: Separate key: ResRobot route planner / stop lookup has its own quota tier.
    trafiklab_resrobot_key: str | None = None

    gtfs_url: str = "https://api.resrobot.se/gtfs/sweden.zip"
    resrobot_base: str = "https://api.resrobot.se/v2.1"
    flixbus_base: str = "https://global.api.flixbus.com"
    freerider_base: str = "https://www.hertzfreerider.se"

    #: Self-hosted OSRM for Freerider road legs. Unset -> great-circle fallback.
    osrm_base: str | None = None

    # --- Storage ------------------------------------------------------------
    data_dir: Path = DATA_DIR
    db_path: Path = DATA_DIR / "tripps.sqlite3"
    gtfs_zip_path: Path = DATA_DIR / "sweden.zip"

    # --- Behaviour ----------------------------------------------------------
    budget: PricingBudget = Field(default_factory=PricingBudget)
    ttl: CacheTTL = Field(default_factory=CacheTTL)

    http_timeout_seconds: float = 20.0
    user_agent: str = "tripps/0.1 (personal trip planner; +https://github.com/local/tripps)"

    #: Freerider inventory poll interval. Community pollers run at 10s; we are a
    #: planner, not a sniper, so we stay well back from that.
    freerider_poll_seconds: int = 300

    #: Background maintenance interval: re-run price-source canaries, recalibrate the routing
    #: floors from newly logged fares, and reload them into the live router. 0 disables it.
    scheduler_seconds: int = 6 * 3600

    #: A GTFS feed older than this reads as stale on /health - it should be re-fetched daily.
    feed_stale_hours: int = 48

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Test hook: drop the memoized Settings so env changes take effect."""
    global _settings
    _settings = None
