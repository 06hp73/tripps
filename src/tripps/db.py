"""SQLite persistence: quote cache, Freerider offer log, adapter health.

The quote cache is what makes the fan-out budget affordable and what lets the planner
answer at all when an unofficial endpoint is down: an expired entry is still served,
marked STALE, instead of the leg going unpriced.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .models import Leg, PriceConfidence, Quote

SCHEMA = """
CREATE TABLE IF NOT EXISTS quote_cache (
    cache_key   TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    amount_ore  INTEGER,
    currency    TEXT NOT NULL DEFAULT 'SEK',
    confidence  TEXT NOT NULL,
    fare_class  TEXT,
    deeplink    TEXT,
    note        TEXT,
    fetched_at  TEXT NOT NULL,
    ttl_seconds INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quote_source ON quote_cache(source);
CREATE INDEX IF NOT EXISTS idx_quote_fetched ON quote_cache(fetched_at);

-- Append-only record of Freerider inventory, so we can measure real offer churn
-- (an open question the plan flags) and serve last-good inventory if the endpoint dies.
CREATE TABLE IF NOT EXISTS freerider_offer (
    offer_id       INTEGER NOT NULL,
    route_id       INTEGER NOT NULL,
    seen_at        TEXT NOT NULL,
    pickup_trac    TEXT NOT NULL,
    return_trac    TEXT NOT NULL,
    available_at   TEXT NOT NULL,
    latest_return  TEXT NOT NULL,
    expire_time    TEXT,
    car_model      TEXT,
    distance_km    REAL,
    travel_minutes INTEGER,
    payload        TEXT NOT NULL,
    PRIMARY KEY (route_id, seen_at)
);
CREATE INDEX IF NOT EXISTS idx_fr_seen ON freerider_offer(seen_at);
CREATE INDEX IF NOT EXISTS idx_fr_route ON freerider_offer(route_id);

CREATE TABLE IF NOT EXISTS adapter_health (
    name       TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    detail     TEXT,
    checked_at TEXT NOT NULL
);

-- Generic TTL cache for payloads that are not per-leg quotes: a day's flight offers for
-- an airport pair, the last-good Freerider snapshot, and so on. Expired rows are kept so
-- a dead upstream can still be answered from stale data, explicitly marked as such.
CREATE TABLE IF NOT EXISTS kv_cache (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    ttl_seconds INTEGER NOT NULL
);

-- Phase-2 reprice deltas, used to calibrate the routing price floors (plan risk #8).
CREATE TABLE IF NOT EXISTS reprice_delta (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    source      TEXT NOT NULL,
    mode        TEXT NOT NULL,
    operator    TEXT,
    distance_km REAL,
    floor_ore   INTEGER NOT NULL,
    actual_ore  INTEGER NOT NULL
);

-- A user's standing interest in a Freerider route. The poller matches live inventory against
-- these and records a hit the first time a matching free car appears.
CREATE TABLE IF NOT EXISTS freerider_watch (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    origin      TEXT NOT NULL,
    destination TEXT NOT NULL,
    origin_lat  REAL NOT NULL,
    origin_lon  REAL NOT NULL,
    dest_lat    REAL NOT NULL,
    dest_lon    REAL NOT NULL,
    radius_km   REAL NOT NULL,
    webhook_url TEXT,
    created_at  TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1
);

-- One row per (watch, car) so a car is announced once, not on every poll.
CREATE TABLE IF NOT EXISTS freerider_hit (
    watch_id     INTEGER NOT NULL,
    route_id     INTEGER NOT NULL,
    seen_at      TEXT NOT NULL,
    pickup       TEXT,
    dropoff      TEXT,
    available_at TEXT,
    car          TEXT,
    PRIMARY KEY (watch_id, route_id)
);
CREATE INDEX IF NOT EXISTS idx_hit_seen ON freerider_hit(seen_at);

-- Travel cards (regional period tickets) the user holds; covered legs price at 0.
CREATE TABLE IF NOT EXISTS travel_card (
    provider_id TEXT PRIMARY KEY,
    added_at    TEXT NOT NULL
);

-- Second-hand period tickets rented via tickital: a provider's card, valid only over a date
-- window, at a stated rental price. Covered legs price at 0 within the window (see passes.py).
CREATE TABLE IF NOT EXISTS tickital_rental (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    price_ore   INTEGER NOT NULL,
    valid_from  TEXT NOT NULL,
    valid_to    TEXT NOT NULL,
    note        TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rental_window ON tickital_rental(valid_from, valid_to);

-- Per-operator price floors calibrated from reprice_delta, replacing the hand-set defaults.
-- These feed back into the router's price lower bound: tighter (but still safe) floors mean
-- a smaller Pareto frontier and fewer upstream price calls per search.
CREATE TABLE IF NOT EXISTS operator_floor (
    operator     TEXT PRIMARY KEY,
    mode         TEXT NOT NULL,
    base_ore     INTEGER NOT NULL,
    per_km_ore   INTEGER NOT NULL,
    samples      INTEGER NOT NULL,
    updated_at   TEXT NOT NULL
);
"""


def leg_cache_key(source: str, leg: Leg) -> str:
    """Stable identity for "this exact departure, priced by this source".

    service_ref is included because two operators can run the same o/d at the same
    minute, and the price belongs to the trip, not the timeslot.
    """
    raw = "|".join(
        [
            source,
            leg.mode.value,
            leg.operator or "",
            leg.from_stop.id,
            leg.to_stop.id,
            leg.departure.isoformat(),
            leg.service_ref or "",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Database:
    """Thin synchronous SQLite wrapper.

    SQLite writes are serialized behind a lock rather than opened per-thread: the
    workload is a handful of writes per search, and a single connection with WAL keeps
    the concurrent-reader story simple.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # --- quote cache ------------------------------------------------------

    def get_quote(self, source: str, leg: Leg, *, now: datetime | None = None) -> Quote | None:
        """Return a cached quote, downgrading its confidence as it ages.

        Fresh -> CACHED. Past TTL -> STALE (still returned; the caller decides whether
        to refresh, and we would rather show a stale number with a warning than claim
        the leg has no price).
        """
        now = now or datetime.now(UTC)
        key = leg_cache_key(source, leg)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM quote_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None

        fetched_at = datetime.fromisoformat(row["fetched_at"])
        age = (now - fetched_at).total_seconds()
        stored = PriceConfidence(row["confidence"])

        if stored is PriceConfidence.UNAVAILABLE:
            # Do not let a cached failure mask a source that has since recovered.
            if age > 60:
                return None
            confidence = PriceConfidence.UNAVAILABLE
        elif stored is PriceConfidence.ESTIMATED:
            confidence = PriceConfidence.ESTIMATED  # fare tables do not go stale like quotes
        elif age > row["ttl_seconds"]:
            confidence = PriceConfidence.STALE
        else:
            confidence = PriceConfidence.CACHED

        return Quote(
            source=row["source"],
            amount_ore=row["amount_ore"],
            currency=row["currency"],
            confidence=confidence,
            fare_class=row["fare_class"],
            deeplink=row["deeplink"],
            note=row["note"],
            fetched_at=fetched_at,
            ttl_seconds=row["ttl_seconds"],
        )

    def put_quote(self, source: str, leg: Leg, quote: Quote, ttl_seconds: int) -> None:
        key = leg_cache_key(source, leg)
        fetched_at = (quote.fetched_at or datetime.now(UTC)).isoformat()
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO quote_cache
                    (cache_key, source, amount_ore, currency, confidence,
                     fare_class, deeplink, note, fetched_at, ttl_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    amount_ore=excluded.amount_ore,
                    currency=excluded.currency,
                    confidence=excluded.confidence,
                    fare_class=excluded.fare_class,
                    deeplink=excluded.deeplink,
                    note=excluded.note,
                    fetched_at=excluded.fetched_at,
                    ttl_seconds=excluded.ttl_seconds
                """,
                (
                    key,
                    source,
                    quote.amount_ore,
                    quote.currency,
                    quote.confidence.value,
                    quote.fare_class,
                    quote.deeplink,
                    quote.note,
                    fetched_at,
                    ttl_seconds,
                ),
            )

    def purge_quotes(self, older_than: timedelta) -> int:
        cutoff = (datetime.now(UTC) - older_than).isoformat()
        with self._write() as conn:
            cur = conn.execute("DELETE FROM quote_cache WHERE fetched_at < ?", (cutoff,))
            return cur.rowcount

    # --- freerider log ----------------------------------------------------

    def log_freerider_offers(self, offers: list[dict], seen_at: datetime | None = None) -> int:
        """Record one inventory snapshot. Returns rows written."""
        seen = (seen_at or datetime.now(UTC)).isoformat()
        rows = [
            (
                o["offer_id"],
                o["route_id"],
                seen,
                o["pickup_trac"],
                o["return_trac"],
                o["available_at"],
                o["latest_return"],
                o.get("expire_time"),
                o.get("car_model"),
                o.get("distance_km"),
                o.get("travel_minutes"),
                json.dumps(o.get("payload", {}), ensure_ascii=False),
            )
            for o in offers
        ]
        with self._write() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO freerider_offer
                    (offer_id, route_id, seen_at, pickup_trac, return_trac,
                     available_at, latest_return, expire_time, car_model,
                     distance_km, travel_minutes, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def latest_freerider_snapshot(self) -> list[dict]:
        """Last-good inventory: every route from the most recent snapshot timestamp."""
        with self._lock:
            row = self._conn.execute("SELECT MAX(seen_at) AS t FROM freerider_offer").fetchone()
            if row is None or row["t"] is None:
                return []
            rows = self._conn.execute(
                "SELECT payload FROM freerider_offer WHERE seen_at = ?", (row["t"],)
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    # --- generic ttl cache ------------------------------------------------

    def cache_put(self, key: str, value: object, ttl_seconds: int) -> None:
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO kv_cache (key, value, fetched_at, ttl_seconds)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    fetched_at=excluded.fetched_at,
                    ttl_seconds=excluded.ttl_seconds
                """,
                (key, json.dumps(value, ensure_ascii=False), datetime.now(UTC).isoformat(), ttl_seconds),
            )

    def cache_get(self, key: str, *, allow_stale: bool = False) -> tuple[object, bool] | None:
        """Return (value, is_stale), or None when absent.

        Stale data is offered only when the caller asks. A stale flight price is worth
        showing with a warning; a stale one silently substituted for a fresh one is not.
        """
        with self._lock:
            row = self._conn.execute("SELECT * FROM kv_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        age = (datetime.now(UTC) - datetime.fromisoformat(row["fetched_at"])).total_seconds()
        stale = age > row["ttl_seconds"]
        if stale and not allow_stale:
            return None
        return json.loads(row["value"]), stale

    # --- health & calibration --------------------------------------------

    def set_health(self, name: str, state: str, detail: str = "") -> None:
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO adapter_health (name, state, detail, checked_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    state=excluded.state, detail=excluded.detail, checked_at=excluded.checked_at
                """,
                (name, state, detail, datetime.now(UTC).isoformat()),
            )

    def get_health(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT name, state FROM adapter_health").fetchall()
        return {r["name"]: r["state"] for r in rows}

    def record_reprice_delta(
        self,
        *,
        source: str,
        mode: str,
        operator: str | None,
        distance_km: float | None,
        floor_ore: int,
        actual_ore: int,
    ) -> None:
        """Log (floor, actual) so floors can be calibrated from real data, not guesses."""
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO reprice_delta
                    (recorded_at, source, mode, operator, distance_km, floor_ore, actual_ore)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    source,
                    mode,
                    operator,
                    distance_km,
                    floor_ore,
                    actual_ore,
                ),
            )

    def floor_violations(self) -> list[sqlite3.Row]:
        """Rows where the routing floor exceeded the real price.

        Each of these is a correctness bug: the router may have pruned the true
        cheapest itinerary. Surfaced by the health endpoint and the test suite.
        """
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM reprice_delta WHERE floor_ore > actual_ore ORDER BY recorded_at DESC"
            ).fetchall()

    def reprice_observations(self) -> list[sqlite3.Row]:
        """Every (operator, mode, distance, actual) point, for floor calibration."""
        with self._lock:
            return self._conn.execute(
                "SELECT operator, mode, distance_km, actual_ore FROM reprice_delta "
                "WHERE operator IS NOT NULL AND distance_km > 0 AND actual_ore > 0"
            ).fetchall()

    # --- freerider watches ------------------------------------------------

    def add_watch(
        self,
        *,
        origin: str,
        destination: str,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        radius_km: float,
        webhook_url: str | None = None,
    ) -> int:
        with self._write() as conn:
            cur = conn.execute(
                """
                INSERT INTO freerider_watch
                    (origin, destination, origin_lat, origin_lon, dest_lat, dest_lon,
                     radius_km, webhook_url, created_at, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    origin, destination, origin_lat, origin_lon, dest_lat, dest_lon,
                    radius_km, webhook_url, datetime.now(UTC).isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def list_watches(self, *, active_only: bool = True) -> list[sqlite3.Row]:
        query = "SELECT * FROM freerider_watch"
        if active_only:
            query += " WHERE active = 1"
        with self._lock:
            return self._conn.execute(query + " ORDER BY created_at DESC").fetchall()

    def deactivate_watch(self, watch_id: int) -> bool:
        with self._write() as conn:
            cur = conn.execute(
                "UPDATE freerider_watch SET active = 0 WHERE id = ?", (watch_id,)
            )
            return cur.rowcount > 0

    def record_hit(
        self,
        watch_id: int,
        *,
        route_id: int,
        pickup: str,
        dropoff: str,
        available_at: str,
        car: str,
    ) -> bool:
        """Record a matched car. Returns True only the first time this (watch, car) is seen."""
        with self._write() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO freerider_hit
                    (watch_id, route_id, seen_at, pickup, dropoff, available_at, car)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (watch_id, route_id, datetime.now(UTC).isoformat(), pickup, dropoff, available_at, car),
            )
            return cur.rowcount > 0

    def recent_hits(self, *, limit: int = 50) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM freerider_hit ORDER BY seen_at DESC LIMIT ?", (limit,)
            ).fetchall()

    # --- travel cards -----------------------------------------------------

    def add_card(self, provider_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO travel_card (provider_id, added_at) VALUES (?, ?)",
                (provider_id, datetime.now(UTC).isoformat()),
            )

    def remove_card(self, provider_id: str) -> bool:
        with self._write() as conn:
            cur = conn.execute("DELETE FROM travel_card WHERE provider_id = ?", (provider_id,))
            return cur.rowcount > 0

    def list_cards(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT provider_id FROM travel_card ORDER BY added_at"
            ).fetchall()
        return [r["provider_id"] for r in rows]

    # --- tickital rentals -------------------------------------------------

    def add_tickital_rental(
        self,
        *,
        provider_id: str,
        price_ore: int,
        valid_from: date,
        valid_to: date,
        note: str = "",
    ) -> int:
        with self._write() as conn:
            cur = conn.execute(
                """
                INSERT INTO tickital_rental
                    (provider_id, price_ore, valid_from, valid_to, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    provider_id,
                    price_ore,
                    valid_from.isoformat(),
                    valid_to.isoformat(),
                    note,
                    datetime.now(UTC).isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def remove_tickital_rental(self, rental_id: int) -> bool:
        with self._write() as conn:
            cur = conn.execute("DELETE FROM tickital_rental WHERE id = ?", (rental_id,))
            return cur.rowcount > 0

    def list_tickital_rentals(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM tickital_rental ORDER BY valid_from, id"
            ).fetchall()

    def active_tickital_rentals(self, on_date: date) -> list[sqlite3.Row]:
        day = on_date.isoformat()
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM tickital_rental WHERE valid_from <= ? AND valid_to >= ? "
                "ORDER BY valid_from, id",
                (day, day),
            ).fetchall()

    # --- calibrated floors ------------------------------------------------

    def put_operator_floors(self, floors: list[dict]) -> None:
        """Replace the calibrated-floor table with a fresh batch."""
        with self._write() as conn:
            conn.executemany(
                """
                INSERT INTO operator_floor (operator, mode, base_ore, per_km_ore, samples, updated_at)
                VALUES (:operator, :mode, :base_ore, :per_km_ore, :samples, :updated_at)
                ON CONFLICT(operator) DO UPDATE SET
                    mode=excluded.mode, base_ore=excluded.base_ore,
                    per_km_ore=excluded.per_km_ore, samples=excluded.samples,
                    updated_at=excluded.updated_at
                """,
                floors,
            )

    def get_operator_floors(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT operator, mode, base_ore, per_km_ore, samples FROM operator_floor"
            ).fetchall()
