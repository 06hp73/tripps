"""Domain model for multimodal Swedish trip planning.

Money is represented in integer öre (1 SEK = 100 öre) everywhere. Float money is
never stored or summed: dynamic rail fares and per-leg totals get added together
many times over, and binary floats accumulate error that shows up as 1-öre diffs
between the routed total and the sum of the breakdown.

Times inside the routing core are integer "seconds since the start of the service
day" (see routing.timetable), because GTFS legitimately expresses departures past
midnight as 25:10:00. Times at the API boundary are timezone-aware datetimes in
Europe/Stockholm.
"""

from __future__ import annotations

import enum
import math
from datetime import UTC, datetime

from pydantic import BaseModel, Field, computed_field, field_validator

SEK = 100  # öre per krona
STOCKHOLM_TZ = "Europe/Stockholm"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. An estimate - real track/road distance is longer."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class TransportMode(str, enum.Enum):
    """Modes the planner can put in an itinerary.

    LOCAL_TRANSIT is separated from BUS/TRAIN because we have schedules for it but
    no price source; it is priced as an estimate and flagged, never silently summed
    into a "cheapest" claim as if it were exact.
    """

    TRAIN = "train"
    BUS = "bus"
    FLIGHT = "flight"
    FREERIDER = "freerider"
    FERRY = "ferry"
    LOCAL_TRANSIT = "local_transit"
    WALK = "walk"


#: Estimated grams of CO2e per passenger-km, for the itinerary CO2 figure. Swedish electric
#: rail is near-zero; coach ~30; domestic flight 127 incl. high-altitude uplift; a relocated
#: Freerider car ~170 (single occupant); a ferry is a rough placeholder. Sources in
#: `Itinerary.co2_grams`.
_CO2_G_PER_PKM = {
    TransportMode.TRAIN: 1,
    TransportMode.BUS: 30,
    TransportMode.FLIGHT: 127,
    TransportMode.FREERIDER: 170,
    TransportMode.FERRY: 120,
    TransportMode.LOCAL_TRANSIT: 30,
    TransportMode.WALK: 0,
}


#: Modes whose legs are "long-distance" for the purposes of the intercity network.
LONG_DISTANCE_MODES = frozenset(
    {
        TransportMode.TRAIN,
        TransportMode.BUS,
        TransportMode.FLIGHT,
        TransportMode.FREERIDER,
        TransportMode.FERRY,
    }
)


class PriceConfidence(str, enum.Enum):
    """How much to trust a Quote.

    EXACT      - fetched live from the operator's own sales channel for this departure.
    CACHED     - EXACT, but fetched earlier and still inside its TTL.
    STALE      - past TTL, shown with a warning rather than dropped.
    ESTIMATED  - computed from a fare table or cost model, not a live quote.
    UNAVAILABLE- no price could be obtained; the leg carries a deeplink instead.
    """

    EXACT = "exact"
    CACHED = "cached"
    STALE = "stale"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


#: Confidences that represent a real, summable amount.
PRICED_CONFIDENCES = frozenset(
    {
        PriceConfidence.EXACT,
        PriceConfidence.CACHED,
        PriceConfidence.STALE,
        PriceConfidence.ESTIMATED,
    }
)


class Stop(BaseModel):
    """A place a leg can start or end: GTFS stop, airport, or Freerider station."""

    id: str
    name: str
    lat: float
    lon: float
    modes: frozenset[TransportMode] = frozenset()
    #: Source-native identifiers, e.g. {"gtfs": "740000001", "iata": "ARN",
    #: "freerider_trac": "SWARN01", "flixbus_city": "40dfdbe7-..."}.
    external_ids: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": True}

    @field_validator("lat")
    @classmethod
    def _lat_range(cls, v: float) -> float:
        if not -90.0 <= v <= 90.0:
            raise ValueError(f"latitude out of range: {v}")
        return v

    @field_validator("lon")
    @classmethod
    def _lon_range(cls, v: float) -> float:
        if not -180.0 <= v <= 180.0:
            raise ValueError(f"longitude out of range: {v}")
        return v


class Quote(BaseModel):
    """A price for one leg (or one whole itinerary, for through-fares)."""

    source: str
    amount_ore: int | None
    currency: str = "SEK"
    confidence: PriceConfidence
    fare_class: str | None = None
    deeplink: str | None = None
    fetched_at: datetime | None = None
    ttl_seconds: int | None = None
    note: str | None = None
    #: Set on the leg(s) a tickital rental coupon rewrote (the charged leg and its zeroed
    #: siblings all carry the same id). The stable key that lets a caller summing more than one
    #: itinerary - round trip, fare calendar - charge a one-time period cost once, by identity
    #: rather than by matching an amount two distinct rentals could share.
    coupon_rental_id: int | None = None
    #: A station fee added on top of the fare (Arlanda C passage), kept SEPARATE from
    #: `amount_ore` on purpose: it is applied after pricing, whoever priced the leg, and a
    #: pass or rental does not cover it. Keeping it out of `amount_ore` means the coupon's
    #: `amount_ore == rental.price_ore` identity survives a surcharge landing on the charged
    #: leg. It IS part of the itinerary total (see `Itinerary.total_price_ore`).
    surcharge_ore: int | None = None
    #: An advisory (not a re-price): if buying two separate tickets split at a major hub is
    #: firmly cheaper than this through fare, the human-readable saving is recorded here. It
    #: does NOT change `amount_ore` or the itinerary total - the authoritative price stays the
    #: bookable through fare - because a split is two contracts with no rebooking protection
    #: across the break, an honest tip rather than the number we stand behind.
    split_hint: str | None = None

    @field_validator("amount_ore", "surcharge_ore")
    @classmethod
    def _non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError(f"negative price: {v}")
        return v

    @property
    def total_ore(self) -> int | None:
        """The fare plus any surcharge, or None if the fare itself is unpriced."""
        if self.amount_ore is None:
            return None
        return self.amount_ore + (self.surcharge_ore or 0)

    @property
    def is_priced(self) -> bool:
        return self.amount_ore is not None and self.confidence in PRICED_CONFIDENCES

    @property
    def amount_sek(self) -> float | None:
        """Display only. Never sum these."""
        return None if self.amount_ore is None else self.amount_ore / SEK

    @classmethod
    def unavailable(cls, source: str, deeplink: str | None = None, note: str | None = None) -> Quote:
        return cls(
            source=source,
            amount_ore=None,
            confidence=PriceConfidence.UNAVAILABLE,
            deeplink=deeplink,
            note=note,
        )


class Leg(BaseModel):
    """One continuous ride (or walk) between two stops."""

    from_stop: Stop
    to_stop: Stop
    mode: TransportMode
    operator: str | None = None
    departure: datetime
    arrival: datetime
    #: Opaque handle back to the source: GTFS trip_id, Freerider offer id, flight number.
    service_ref: str | None = None
    headsign: str | None = None
    quote: Quote | None = None
    #: Every stop this leg calls at, from board to alight inclusive (GTFS ids). Used to check
    #: whether the union of held travel cards covers the *whole* path of a through-train, not
    #: just its endpoints. In-process only - excluded from serialization to keep the API lean.
    via_stop_ids: tuple[str, ...] = Field(default=(), exclude=True, repr=False)
    #: Departure and arrival wall-clock times at each stop in `via_stop_ids`, index-aligned.
    #: Let a caller reconstruct a sub-leg board X -> alight Y of this same physical train (its
    #: departure at X, arrival at Y) to price a split ticket. In-process only, not serialized.
    via_departures: tuple[datetime, ...] = Field(default=(), exclude=True, repr=False)
    via_arrivals: tuple[datetime, ...] = Field(default=(), exclude=True, repr=False)
    #: Ridden polyline length in km: the sum of the route's per-segment distances from board to
    #: alight - the SAME distance McRAPTOR integrated its per-km price floor over. The floor
    #: audit and calibration must use this, never the shorter endpoint straight-line: a per-km
    #: rate fit on the chord but applied over the curve exceeds the true fare on winding routes,
    #: silently breaking the floor <= true_price contract. None when the leg was not built by
    #: the router (walks, synthetic injections). In-process only, not serialized.
    path_km: float | None = Field(default=None, exclude=True, repr=False)

    @field_validator("arrival")
    @classmethod
    def _arrival_after_departure(cls, v: datetime, info) -> datetime:
        dep = info.data.get("departure")
        if dep is not None and v < dep:
            raise ValueError(f"arrival {v} precedes departure {dep}")
        return v

    @property
    def duration_seconds(self) -> int:
        return int((self.arrival - self.departure).total_seconds())

    @property
    def is_transit(self) -> bool:
        """Does boarding this leg count as using a vehicle (i.e. a transfer)?"""
        return self.mode is not TransportMode.WALK


class Itinerary(BaseModel):
    """A complete door-to-door option, priced or partially priced."""

    legs: list[Leg]
    warnings: list[str] = Field(default_factory=list)
    #: The router's price lower bound for this journey, carried through from McRAPTOR.
    #: Used to decide which candidates are worth spending an upstream price call on, and
    #: to detect floors that exceeded the truth. None once it is no longer meaningful.
    floor_price_ore: int | None = None

    @field_validator("legs")
    @classmethod
    def _non_empty(cls, v: list[Leg]) -> list[Leg]:
        if not v:
            raise ValueError("itinerary must have at least one leg")
        return v

    # `computed_field` so these appear in `model_dump()` / the JSON API, not just on the
    # Python object - the web UI and any API consumer need the total and the times, which are
    # derived, not stored.
    @computed_field
    @property
    def departure(self) -> datetime:
        return self.legs[0].departure

    @computed_field
    @property
    def arrival(self) -> datetime:
        return self.legs[-1].arrival

    @computed_field
    @property
    def duration_seconds(self) -> int:
        # Wall-nominal by design: same-tzinfo subtraction ignores UTC offsets, so this is the
        # clock-face difference, consistent with every displayed time. On a DST-change night it
        # differs from true elapsed by an hour; SearchConstraints.permits enforces max_duration
        # on the TRUE elapsed instead.
        return int((self.arrival - self.departure).total_seconds())

    @computed_field
    @property
    def transfers(self) -> int:
        """Number of vehicle boardings after the first. Walks are not boardings."""
        boardings = sum(1 for leg in self.legs if leg.is_transit)
        return max(0, boardings - 1)

    @property
    def modes(self) -> list[TransportMode]:
        return [leg.mode for leg in self.legs]

    @property
    def is_multimodal(self) -> bool:
        return len({m for m in self.modes if m is not TransportMode.WALK}) > 1

    @property
    def fully_priced(self) -> bool:
        """True only if every vehicle leg carries a real amount."""
        return all(
            leg.quote is not None and leg.quote.is_priced
            for leg in self.legs
            if leg.mode is not TransportMode.WALK
        )

    @computed_field
    @property
    def total_price_ore(self) -> int | None:
        """Sum of leg prices, or None if any vehicle leg is unpriced.

        Returning None rather than a partial sum is deliberate: a partial sum sorted
        against complete sums would rank an itinerary as "cheapest" precisely because
        we failed to price part of it.
        """
        if not self.fully_priced:
            return None
        return sum(
            leg.quote.amount_ore + (leg.quote.surcharge_ore or 0)
            for leg in self.legs
            if leg.mode is not TransportMode.WALK and leg.quote is not None
        )

    @computed_field
    @property
    def total_price_sek(self) -> float | None:
        total = self.total_price_ore
        return None if total is None else total / SEK

    @computed_field
    @property
    def co2_grams(self) -> int:
        """Rough well-to-wheel CO2e for the journey, grams, over great-circle leg distances.

        Per passenger-km factors (klimatsmartsemester.se / UIC EcoPassenger, 2026): Swedish
        electric rail ~1 g, long-distance coach ~30 g, domestic flight ~127 g CO2e (incl. the
        1.7x high-altitude uplift), a relocated car ~170 g. An estimate, not a certified figure
        - distances are great-circle, so the real number is a little higher.
        """
        total = 0.0
        for leg in self.legs:
            km = haversine_km(
                leg.from_stop.lat, leg.from_stop.lon, leg.to_stop.lat, leg.to_stop.lon
            )
            total += km * _CO2_G_PER_PKM.get(leg.mode, 0)
        return round(total)

    @computed_field
    @property
    def price_confidence(self) -> PriceConfidence:
        """Weakest confidence across vehicle legs: a chain is as strong as its worst link."""
        order = [
            PriceConfidence.UNAVAILABLE,
            PriceConfidence.ESTIMATED,
            PriceConfidence.STALE,
            PriceConfidence.CACHED,
            PriceConfidence.EXACT,
        ]
        confs = [
            leg.quote.confidence if leg.quote else PriceConfidence.UNAVAILABLE
            for leg in self.legs
            if leg.mode is not TransportMode.WALK
        ]
        if not confs:
            return PriceConfidence.EXACT
        return min(confs, key=order.index)


class SearchConstraints(BaseModel):
    """User limits. Price is the objective; these bound the feasible set."""

    max_duration_seconds: int | None = None
    max_transfers: int | None = None
    earliest_departure: datetime | None = None
    latest_arrival: datetime | None = None
    allowed_modes: frozenset[TransportMode] = frozenset(TransportMode)
    #: Freerider requires a driving licence and a Hertz account; let users exclude it.
    include_freerider: bool = True

    def permits(self, itin: Itinerary) -> bool:
        if self.max_duration_seconds is not None:
            # Enforce on TRUE elapsed time. The displayed `duration_seconds` is wall-nominal
            # by design (same-tzinfo subtraction ignores UTC offsets), which under-counts by
            # an hour across the autumn DST fall-back - a 10h30m overnight journey would
            # sneak past a 10h cap. Aware datetimes normalized to UTC measure the real gap;
            # naive ones (tests, fixtures) subtract identically either way.
            dep, arr = itin.departure, itin.arrival
            if dep.tzinfo is not None and arr.tzinfo is not None:
                elapsed = int((arr.astimezone(UTC) - dep.astimezone(UTC)).total_seconds())
            else:
                elapsed = itin.duration_seconds
            if elapsed > self.max_duration_seconds:
                return False
        if self.max_transfers is not None and itin.transfers > self.max_transfers:
            return False
        if self.earliest_departure is not None and itin.departure < self.earliest_departure:
            return False
        if self.latest_arrival is not None and itin.arrival > self.latest_arrival:
            return False
        for leg in itin.legs:
            if leg.mode not in self.allowed_modes:
                return False
            if leg.mode is TransportMode.FREERIDER and not self.include_freerider:
                return False
        return True


class SearchRequest(BaseModel):
    origin: str
    destination: str
    date: str  # ISO date, YYYY-MM-DD, in Europe/Stockholm
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    max_results: int = 5


class SearchResponse(BaseModel):
    origin: Stop
    destination: Stop
    date: str
    itineraries: list[Itinerary]
    warnings: list[str] = Field(default_factory=list)
    #: Per-adapter health so the UI can say "SJ prices unavailable" rather than lie.
    source_status: dict[str, str] = Field(default_factory=dict)
