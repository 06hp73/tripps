"""Runtime configuration.

Every external dependency is optional at import time. The planner must start, and
degrade honestly, when a key is missing or a host is down; only the features that
truly need a credential get disabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import PassengerCategory, TransportMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_data_dir() -> Path:
    """Where the feed, the timetable cache and the database live.

    Order matters, and each branch exists for a real deployment:

    1. `TRIPPS_DATA_DIR` — an explicit choice always wins. Docker sets it.
    2. `<checkout>/data` when the package sits in a source tree (`.../src/tripps/config.py`
       with a `pyproject.toml` two levels up). A developer's state stays in their checkout,
       which is what every existing install expects.
    3. The platform's user data directory. This is the packaged-app case: a `.app` may live
       in /Applications, be quarantined, be replaced wholesale on update, or be read-only, so
       writing 68 MB of feed and a SQLite database *inside the bundle* is wrong. It also frees
       the Docker image from the editable-install requirement, since nothing depends any more
       on the package's own path.
    """
    env = os.environ.get("TRIPPS_DATA_DIR")
    if env:
        return Path(env).expanduser()

    if (PROJECT_ROOT / "pyproject.toml").is_file():
        return PROJECT_ROOT / "data"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "tripps"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "tripps"


DATA_DIR = _default_data_dir()


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
    #: After ranking, probe whether an SJ through fare is beaten by two tickets split at a major
    #: hub, and annotate the leg with the saving. Costs a few extra SJ price calls on the shown
    #: itineraries only (bounded by the same per-source budget); the number tripps stands behind
    #: is unchanged. Off-switch for when those extra calls are unwanted.
    enable_split_tickets: bool = True

    #: What a *refine* pass multiplies the per-source call allowance by. A first search hides
    #: itineraries whose legs it ran out of calls to price; asked to look again, it re-prices
    #: the same candidates with this much more allowance. Measured on Uppsala->Göteborg
    #: (2026-08-10): 17 hidden at the base allowance, 5 at double, and nothing improves beyond
    #: that - past 2x the budget stops being what binds.
    refine_call_multiplier: int = 2
    #: Wall-clock ceiling for a refine pass. Higher than `phase2_timeout_seconds` because it
    #: prices strictly more legs; the fares the first pass already fetched come from the quote
    #: cache, so the extra time is spent on the legs that were starved, not on repeats.
    refine_timeout_seconds: float = 90.0

    model_config = SettingsConfigDict(env_prefix="TRIPPS_BUDGET_")

    def refined(self) -> PricingBudget:
        """This budget, with the allowances a refine pass runs on."""
        return self.model_copy(
            update={
                "max_calls_per_source_per_search": (
                    self.max_calls_per_source_per_search * self.refine_call_multiplier
                ),
                "phase2_timeout_seconds": self.refine_timeout_seconds,
            }
        )


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

    #: Who is travelling, when a search does not say. Someone who is a student every day
    #: should not have to pass --passenger on every command.
    passenger: PassengerCategory = PassengerCategory.ADULT
    #: Age the default category is priced at. None uses that category's representative age.
    passenger_age: int | None = None

    #: Freerider inventory poll interval. Community pollers run at 10s; we are a
    #: planner, not a sniper, so we stay well back from that.
    freerider_poll_seconds: int = 300

    #: Background maintenance interval: re-run price-source canaries, recalibrate the routing
    #: floors from newly logged fares, and reload them into the live router. 0 disables it.
    scheduler_seconds: int = 6 * 3600

    #: A GTFS feed older than this reads as stale on /health - it should be re-fetched daily.
    feed_stale_hours: int = 48

    #: A canary result older than this no longer describes the live source, so /health and the
    #: status page mark it stale rather than presenting it as current. It is also the catch-up
    #: threshold: a server starting with rows older than this re-probes before the first
    #: scheduled maintenance pass, instead of serving a dead reading for `scheduler_seconds`.
    canary_stale_hours: int = 24

    @model_validator(mode="after")
    def _follow_data_dir(self) -> Settings:
        """Keep the feed and the database inside `data_dir` when it moves.

        Both defaults are built from the module-level DATA_DIR at class-definition time, so
        setting TRIPPS_DATA_DIR alone used to move the directory and nothing in it: the app
        created the new directory and went on reading the old one. Anything the caller set
        explicitly is left alone — an absolute TRIPPS_DB_PATH outside the data dir stays put.
        """
        if "data_dir" in self.model_fields_set:
            if "db_path" not in self.model_fields_set:
                self.db_path = self.data_dir / "tripps.sqlite3"
            if "gtfs_zip_path" not in self.model_fields_set:
                self.gtfs_zip_path = self.data_dir / "sweden.zip"
        return self

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
