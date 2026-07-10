# tripps

Finds the **cheapest** way to travel between two points **within Sweden**, combining
long-distance train, coach, domestic flight, ferry, and **Hertz Freerider** free one-way
rental cars — including mixed-mode journeys such as *train to Uppsala, free Hertz car to
Arlanda*, or *bus to Kiruna airport, free car to Luleå, night train to Stockholm*.

```
$ tripps search "Stockholm Centralstation" "Göteborg Centralstation" --date 2026-07-13

 * 1. 22:55-05:45 (6h50m, 0 transfers) bus: 420 SEK [exact]
   2. 15:05-21:35 (6h30m, 0 transfers) bus: 570 SEK [exact]
   3. 07:30-13:45 (6h15m, 0 transfers) bus: 800 SEK [exact]
```

## The central problem

**Schedules and prices come from completely different places, and no source gives both.**

Trafiklab's national GTFS feed has every timetable in Sweden and not one fare. Swedish
long-distance rail is yield-managed and sold only through each operator's own channel. So
the system is two phases, the shape Entur uses:

1. **Route** on schedules with a *lower bound* on price (custom McRAPTOR).
2. **Price** the surviving candidates for real, through per-operator adapters, and re-rank.

### Why a custom router

No open-source engine optimizes a dynamically-priced ticket *during* the search. OTP2
computes fares from finished itineraries; MOTIS/nigiri attaches them afterwards with
compile-time-fixed criteria; R5's in-routing fare McRAPTOR ships only static rule-based
regional calculators that cannot express an SJ fare. Optimizing price only *after* routing
means the cheapest-but-slower journey is often never generated to be priced.

At this scale — 1318 stops, ~6900 intercity trips/day — a bespoke McRAPTOR runs a query in
well under a second, so the engine costs less than the fork would.

### The invariant everything rests on

```
floor(leg) <= true_price(leg),  always
```

The router optimizes a bound. If the bound ever exceeds the truth, McRAPTOR can prune the
genuinely cheapest journey before anyone prices it. Every (floor, actual) pair is logged to
`reprice_delta`; violations surface in `/health` and as a search warning rather than
silently corrupting the answer.

### Departure time is a Pareto criterion

FlixBus Stockholm→Göteborg costs 800 SEK at 07:30 and 420 SEK at 22:55 on the same day.
The routing floor is derived from distance and operator, so it is *identical* for both. With
only `(arrival, price)` as criteria the night bus is "later arrival, same price" and gets
pruned — and the planner confidently reports 800 SEK as the cheapest way across Sweden.

So the search is a **range (profile) query**: `(arrival, price, departure)`, with departure
maximized. See `tests/test_profile_query.py`, which locks this shut.

## Data sources

| Source | Gives | Access | Confidence |
|---|---|---|---|
| GTFS Sweden (`api.resrobot.se/gtfs/sweden.zip`) | all national schedules, 49 operators | free, CC0, no key needed today | authoritative |
| FlixBus (`global.api.flixbus.com`) | real coach fares | unofficial, unauthenticated | `exact` |
| SJ (`prod-api.adp.sj.se`) | real train fares (yield-managed) | unofficial, key from site bundle | `exact` |
| Tora / Trainplanet (`wl.tora.trainplanet.com/v1/offers`) | Öresundståg, Mälartåg, Skånetrafiken, Tåg i Bergslagen, Vy Bus4You fares | unofficial, TLS-fingerprint WAF (needs `primp`) | `exact` |
| Hertz Freerider (`hertzfreerider.se/api/transport-routes/`) | free-car inventory | unofficial, unauthenticated | `estimated` |
| Google Flights via `fast-flights` | domestic flight fares | scrape, needs EU consent cookie | `exact` |
| remaining operators (Y-buss, Härjedalingen, local transit …) | — | no obtainable price API | `unavailable` + booking link |

Notes that cost real debugging time, preserved here so nobody repeats them:

- FlixBus wants `departure_date` as `DD.MM.YYYY`; an ISO date returns HTTP 400. The price a
  passenger pays is `price.total_with_platform_fee`, not `price.total`.
- SJ's booking backend authenticates with a subscription key baked into the site's JS
  bundle next to `"Ocp-Apim-Subscription-Key"`; the adapter extracts every candidate and
  keeps the one a probe call accepts, so a rotated key is picked up automatically. Its
  station codes are UIC codes, which are exactly the GTFS stop ids (`740000001` is
  Stockholm Central in both), so a routed leg's ids go straight into the API. Pricing is a
  three-call chain: search, then departures, then per-departure offers; `priceFrom.price`
  is the cheapest fare. Only direct (zero-change) departures are matched, since a routed
  leg is a single train.
- Öresundståg, Mälartåg, Skånetrafiken (Pågatåg), Tåg i Bergslagen and the Turnit coach
  Vy Bus4You all sell through one backend: Trainplanet's Tora white-label. A single
  `POST /v1/offers` returns journeys from *every* operator Trainplanet resells for a pair
  (one live Malmö→Göteborg response carried Öresundståg, SJ and Vy Bus4You), so the adapter
  queries once per O/D and picks the journey whose carrier matches the routed leg. Place
  ids are `urn:x_swe:stn:{UIC}` — the GTFS stop id again. There is no auth, but a
  TLS-fingerprint WAF returns 403 to any non-browser client: a plain httpx POST fails while
  the same bytes from Chrome succeed. So this one adapter speaks through `primp` with
  browser impersonation. SJ and FlixBus are deliberately not routed here — their own
  adapters reach the operator without a reseller in between.
- Freerider's `distance` is the **included mileage allowance** (measured at exactly 1.20 ×
  `originalDistance` across every live route), not the distance you drive. `originalDistance`
  is the drive. A contract test fails if that ratio ever drifts.
- Freerider's `?country=SWEDEN` filters by *destination*: a Kirkenes→Stockholm car appears in
  the Swedish feed. Both endpoints are checked.
- Freerider timestamps are naive local time. Reading them as UTC shifts every pickup window.
- Google Flights serves EU visitors a `consent.google.com` interstitial; without a consent
  cookie `fast-flights` dies on a missing `<script>`. It also returns Frankfurt and Helsinki
  connections for a Stockholm→Göteborg search, which are not journeys within Sweden.
- GTFS `transfers.txt` is directional, but a footpath is not: unmirrored, the router can walk
  from the station to the coach terminal and never walk back.
- Regional trains (`route_type` 106: Öresundståg, Mälartåg) are frequently the *cheapest*
  option on their corridors. A "long-distance only" filter silently hides the cheapest answer.

## Honesty rules

The planner would rather say nothing than say something false.

- A leg it cannot price gets `unavailable` and a booking link. The itinerary's total becomes
  `None` and ranks **below** every fully priced option — a hole must never sum to zero and
  masquerade as a bargain.
- Freerider quotes are always `estimated`, never `exact`. Hertz publishes neither the tank
  range nor the excess-mileage rate anywhere machine-readable, so a non-zero Freerider price
  is *our cost model's assumption*. Presenting it as fact would be the most misleading thing
  this program could do, because Freerider is exactly the mode people choose because it looks
  free. Short trips are genuinely free; a 713 km car costs ~200 SEK in fuel.
- Booking a Freerider car is not automated. Listing is public; reserving needs the user's own
  Hertz login. The UI surfaces a deep link and the return deadline.

## Install and run

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev,flights]"

.venv/bin/tripps fetch-gtfs                       # ~67 MB, CC0, no key required
.venv/bin/tripps freerider                        # list today's free cars
.venv/bin/tripps search "Stockholm Centralstation" "Göteborg Centralstation" \
    --date 2026-07-13 --limit 5
.venv/bin/tripps search "Kiruna" "Stockholm Centralstation" --date 2026-07-13
.venv/bin/tripps serve                            # web UI + JSON API on :8000
```

`--flights` enables the Google Flights scrape (slow, and gray under Google's ToS).
Everything is configurable through `TRIPPS_*` environment variables; see `config.py`.

Endpoints: `GET /` (UI), `POST /search`, `GET /api/search`, `GET /api/stops`,
`GET /api/freerider`, `GET /health`.

## Layout

```
src/tripps/
  models.py            Stop, Leg, Itinerary, Quote. Money is integer öre, never float.
  timeutil.py          GTFS service-day seconds <-> wall clock.
  routing/
    timetable.py       RAPTOR arrays; per-search synthetic-route overlay.
    mcraptor.py        the search: (arrival, price, departure) Pareto, range query.
    floors.py          price lower bounds. Deliberately too low, never too high.
    synthetic.py       Freerider offers -> scheduled trips. No algorithm changes needed.
    journey.py         labels -> itineraries; candidate spread and collapse.
  ingest/              gtfs, freerider, flights, airports (OurAirports-derived)
  pricing/             flixbus, sj, tora, freerider, flights, operators (link-out), orchestrator
  search.py            Planner: resolve, overlay, route, price, rank.
  api/                 FastAPI + a dependency-free web UI.
```

## Verification

```bash
.venv/bin/python -m pytest tests/ -q     # 148 tests
.venv/bin/ruff check src tests
```

Tests run offline: the FlixBus and Freerider fixtures are recorded live responses, so a
schema change upstream fails a contract test instead of quietly mispricing a leg.

## Known gaps

- **Remaining unpriced operators.** Most rail and coach is now priced (SJ directly;
  Öresundståg, Mälartåg, Skånetrafiken, Tåg i Bergslagen and Vy Bus4You via Tora). What is
  left — Y-buss, Härjedalingen, Masexpressen, Flygbussarna, and pure local transit — either
  sells through a different backend or has no obtainable price source, so those legs are
  routed and shown with a booking link. `StaticFareAdapter` exists to hold published fixed
  fares and ships empty on purpose.
- Local-transit legs are scheduled but never priced, and say so.
- The road matrix falls back to a great-circle estimate unless `TRIPPS_OSRM_BASE` is set.
- Legal: unofficial endpoints and scraping are used here for personal, non-commercial use.
  Systematic extraction and redisplay would need a look at each source's terms and at Swedish
  *katalogskydd* / the EU database right first.
