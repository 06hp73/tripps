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
- A discount is only ever shown where the traveller can actually buy it. FlixBus's search
  endpoint happily answers `products={"student":1}` with 20% off, but shop.flixbus.se sells
  only *Vuxna*, *Barn 0–15 år* and bikes — so tripps quotes FlixBus at the adult fare for
  every category and says so, per leg and once per journey. Ranking on an unbookable fare
  would be a lie in the one number this program exists to get right.

## Refining a search

A search hides itineraries it could not fully price, and most of the time that is not a
missing fare — it is the per-source call allowance running out mid-search. Measured on
Uppsala→Göteborg (2026-08-10, cold cache each run):

| calls/source | hidden | upstream calls | wall clock |
|---|---|---|---|
| 12 (default) | 17 | 55 | 8.9s |
| 24 | 5 | 74 | 14.5s |
| 48 | 5 | 74 | 14.0s |

Past 24 nothing changes: the budget stops being what binds, and the residual 5 have no
purchasable fare at any allowance. So the fix is offered, not applied — `tripps search
--refine`, `GET /api/search?refine=1`, or the "Look again at the N hidden option(s)" button in
the warning box.

Refining re-runs the same query with `refine_call_multiplier`× the allowance. It is cheap the
second time because the first pass's fares are already in the quote cache and cost no budget,
which leaves the whole new allowance for the legs that were starved:

```
step 1 (budget 12, cold)     17 hidden   55 calls
step 2 (budget 24, warm)      5 hidden   28 calls   <- the refine
one shot (budget 24, cold)    5 hidden   74 calls
```

On Lund→Stockholm the two-step even beat the single larger budget (4 hidden vs 6), for the
same reason: cached quotes are free, so a second pass reaches legs one bigger budget still
could not.

**Why it is never automatic.** Across every route and budget measured, the cheapest fare did
not move — 503, 465, 540 SEK regardless. Refining buys completeness, not a cheaper journey, so
spending a second round of calls against undocumented endpoints on every search (including the
ones nobody reads) is a bad trade. A floor-based auto-trigger was tried and rejected: the
routing floors are deliberately pessimistic, so every hidden itinerary's bound sits below the
cheapest shown price (21/21, 6/6, 1/1 on three routes) and the trigger fires always.

## Split ticketing

SJ prices every origin/destination pair independently and allocates cheap seats per segment,
so the through fare is its own bucket rather than the sum of anything. Two tools, for two
different questions.

**During a search** (`split.py`), one break at one of twelve curated hubs is tested on the
itineraries actually shown, and reported as an advisory `split_hint`. Cheap, and it never
moves the price tripps stands behind.

**On demand** (`tripps splits`, or the "scan split tickets" button on any SJ leg), one train
is scanned exhaustively:

```bash
tripps splits "Stockholm Centralstation" "Malmö Centralstation" --date 2026-08-10
tripps splits "Stockholm Centralstation" "Malmö Centralstation" --date 2026-08-10 --departure 07:19
```

```
07:19 Stockholm Centralstation -> 11:53 Malmö Centralstation (SJ 20740523007636)
  through fare: 1735 SEK
  cheapest chain: 1570 SEK in 4 ticket(s)
    07:19 Stockholm Centralstation  -> 09:32 Tranås station             885 SEK
    09:32 Tranås station            -> 09:56 Nässjö Centralstation      185 SEK
    09:57 Nässjö Centralstation     -> 10:30 Alvesta station            305 SEK
    10:31 Alvesta station           -> 11:53 Malmö Centralstation       195 SEK
  => 165 SEK cheaper than the through fare.
```

The optimisation is exact, not greedy: with a fare for every pair of calling points, the
cheapest chain is a shortest path through a DAG, so it finds the best combination over *any*
number of breaks. On that train it beat the best single break (1600 via Alvesta) by breaking
four times, including at Tranås — a station no curated hub list contained.

Why it is a separate command rather than part of every search. The saving lives on
**expensive** departures: measured across Stockholm→Malmö on 2026-08-10, the 1735 SEK peak
trains split down by 115–165 SEK while the 955 SEK cheapest departure of the day cannot be
beaten at all. A cheapest-first search never shows those trains, so the finding is only worth
its cost once the traveller has chosen a departure. One scan is ~3 upstream calls per pair of
calling points — 108 calls for a 9-stop train — which is precisely the fan-out a normal
search must never do.

A chain of tickets is several contracts with no rebooking or delay protection across a break,
so it must beat the through fare by `MIN_SAVING_ORE` before it is called cheaper, and a single
ticket wins a tie. When SJ sells no through fare at all, a complete chain is reported as what
it is: the only way to buy that train. What the scan did *not* cover is always stated, and the
two reasons are kept apart — stops skipped to stay under `--max-points` (raise it and they are
scanned) versus stops where the operator lets nobody board or alight (raising it changes
nothing).

## Passenger categories

`--passenger student` (also `youth`, `senior`, `child`, or `TRIPPS_PASSENGER`) prices the
whole search for that traveller. The discount is *fetched*, never derived: it varies by
operator and by age, so every source is asked for its own number.

Verified live on 2026-08-02, the 05:13 Stockholm C → Göteborg C:

| source | adult | student | senior | child (8) |
|---|---|---|---|---|
| SJ | 515 | 411 | 463 | 437 |
| Tora (Trainplanet) | 530 | 423 | 477 | not sold |
| FlixBus | 510 | not sold | not sold | not sold |

SJ names its own tiers in `/v3/config.passengerCategories` — `ADULT`, `CHILD_AND_YOUTH`
(0–25), `STUDENT` (15–120), `SENIOR` (18–120) — and requires an `age` *inside*
`passengerCategory`, answering `400 Age cannot be null for type STUDENT` otherwise. There is
no separate child tier: a child is a young `CHILD_AND_YOUTH`, and the age carries the
difference. Tora accepts `ADULT`/`STUDENT`/`SENIOR` and rejects `YOUTH`/`CHILD` outright.
`--age` overrides the representative age per tier (student 22, youth 20, senior 70, child 8);
an age outside a tier's span is refused locally rather than as an opaque 400 mid-search.

Two things this touches that are easy to miss:

- **The quote cache key** includes the category *and* the age, because the same seat on the
  same train is 515 SEK for an adult and 411 for a student. Adult keys stay unsuffixed, so
  fares cached before categories existed remain valid.
- **The price floors** are calibrated per `(operator, category)`. An adult floor is fitted to
  sit just under the cheapest *adult* fare, so applying it to a student search would put the
  bound above the fare and let McRAPTOR prune the genuinely cheapest journey — the one failure
  [the invariant](#the-invariant-everything-rests-on) exists to prevent. Until a category has
  its own `MIN_SAMPLES`, it routes on the adult floor scaled by `DISCOUNT_FLOOR_SCALE` (0.5,
  against a deepest observed discount of 0.80), and the violation detector backstops it.

## Install and run

Needs **Python 3.12+** and nothing else — no accounts, no API keys.

```bash
./run.sh
```

That is the whole thing: it builds the virtualenv, installs the package, downloads the
~65 MB timetable feed, starts the server on http://127.0.0.1:8000 and opens it. Each step is
skipped when it is already done, so the second run starts in about a second. Pass anything
through to the CLI instead of serving:

```bash
./run.sh search "Lund C" "Stockholm C" --date 2026-08-09
./run.sh cards providers
TRIPPS_PORT=8010 ./run.sh                         # if 8000 is taken
```

### By hand

The commands below use [uv](https://docs.astral.sh/uv/); if you do not have it,
`python3.12 -m venv .venv` and `.venv/bin/pip install -e ".[dev]"` are equivalent.

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"                        # add ,flights for the Google Flights scrape

.venv/bin/tripps fetch-gtfs                       # ~65 MB, CC0, no key required
.venv/bin/tripps freerider                        # list today's free cars
.venv/bin/tripps search "Stockholm Centralstation" "Göteborg Centralstation" \
    --date 2026-07-13 --limit 5
.venv/bin/tripps search "Kiruna" "Stockholm Centralstation" --date 2026-07-13
.venv/bin/tripps search "Stockholm C" "Göteborg C" --date 2026-07-13 --return 2026-07-16
.venv/bin/tripps search "Stockholm C" "Göteborg C" --modes train,bus   # only these modes
.venv/bin/tripps search "Stockholm C" "Göteborg C" --refine            # fewer options hidden
.venv/bin/tripps search "Stockholm C" "Göteborg C" --passenger student # discounted fares
.venv/bin/tripps search "Stockholm C" "Göteborg C" --passenger child --age 8
.venv/bin/tripps fares  "Stockholm C" "Göteborg C" --start 2026-07-13 --days 7
.venv/bin/tripps serve                            # web UI + JSON API on :8000
```

**Hold a regional period ticket? Register it first — it changes the answer, not the display.**
A covered leg prices at 0, so the cheapest journey itself can differ. The same
Lund→Stockholm search costs 565 SEK with no card and 450 SEK with a Skånetrafiken one,
because the first leg stops being a purchase.

```bash
.venv/bin/tripps cards providers                  # every known regional provider
.venv/bin/tripps cards add skanetrafiken          # …then your own
```

The first search for a given date parses the whole feed (a few seconds to tens of seconds,
logged as it happens) and caches the result, so every later search on that date is fast.

```bash
.venv/bin/tripps canary                           # probe every live price source, alert on drift
.venv/bin/tripps validate                         # route canonical corridors, report pass/warn/fail
.venv/bin/tripps watch add "Uppsala" "Arlanda"    # watch a Freerider route for free cars
.venv/bin/tripps watch poll --interval 300        # …and announce new ones as they appear
```

`--flights` enables the Google Flights scrape (slow, and gray under Google's ToS); it needs
the `flights` extra. Everything else — including all rail and coach pricing — works from the
base install. Everything is configurable through `TRIPPS_*` environment variables; see
`.env.example` for the ones worth knowing and `config.py` for the rest.

`tripps canary` drives each real endpoint (FlixBus, SJ, Tora, Freerider, the GTFS feed) the
way the planner does and asserts a price still comes back, exiting non-zero if any source is
down — run it from cron so a rotated key or reshaped JSON surfaces before users hit it. Its
results also feed `/health` (the `canaries` block) and the live `/api/canary` endpoint.

Every health entry carries its own `checked_at`, `age_label` and `stale` flag, because a
stored `ok` is a claim about the moment it was written, not about now — a server restarted
after three idle weeks would otherwise present three-week-old results as the state of the
sources. Anything older than `TRIPPS_CANARY_STALE_HOURS` (default 24) is greyed out on
`/status` and re-probed once at startup, before the scheduler's first pass. The refresh is
never wired to a request: a page view must not fan out to five undocumented endpoints.

`tripps validate` goes a step further than the canary: it runs a fixed set of canonical
corridors (Stockholm→Göteborg, Malmö→Umeå, a cross-border UL route, an airport-coach route, …)
through the *real* planner and checks invariants — every routable corridor returns an
itinerary, nothing trips a price-floor violation (which would mean the true cheapest was
pruned), and priced totals are sane. It opens with the canary liveness table, exits non-zero
on a hard failure, and — because each corridor is a real search — warms the `reprice_delta`
history that `tripps calibrate` consumes. `GET /status` renders the same operational picture as
an HTML dashboard: live vs scheduled source health, floor violations, and calibration progress.

**Travel cards.** Register a regional period ticket (Skånetrafiken, Västtrafik, SL, …) and
its covered legs price at 0, because a period ticket is unlimited travel in its area. The web
UI has a searchable scroll-list of every provider; the CLI is `tripps cards add/list/remove`
(and `/api/providers`, `/api/cards`). The load-bearing rule, verified against every PTA's own
ticket pages: **the operator name alone is never enough.** The same consortium train
(Öresundståg, Krösatåg, Mälartåg, Tåg i Bergslagen, Norrtåg) is honored by many cards, each
only within its own region, so a card that honors a consortium operator frees a leg only when
*both* endpoints lie in the card's served stops — a Skånetrafiken card frees Öresundståg
Malmö→Lund but not Malmö→Göteborg. Coverage is deliberately conservative: it assumes the
all-zone card variant, never auto-frees SJ (its agency mixes SJ Regional with high-speed),
and prefers to charge over wrongly zeroing a leg. A covered leg is shown, not hidden, at 0
SEK with an "Included with your Skånetrafiken period ticket" note.

**Riding past your card's border.** A period ticket does not stop being valid because the
train carries on. A Hallandstrafiken holder going Halmstad→Göteborg travels free to the edge
of Halland and buys a ticket only for the remainder, so the leg costs the Åsa→Göteborg fare of
**90 SEK**, not the 195 the whole ride costs. tripps prices that remainder as its own
origin/destination pair — a fare in its own right, never a fraction of the through fare — and
does it *during* pricing rather than as a footnote, because it changes the answer: at 195 the
Halmstad coach (160) is the cheapest way to Göteborg, at 90 the train is. The leg carries
"Halmstad Centralstation → Åsa station is covered by your Hallandstrafiken period ticket; only
Åsa station → Göteborg Centralstation is charged."

The reduction is applied to what is *shown*, never to what is *stored*. The quote cache is
keyed on the leg and the traveller, not on which cards happen to be registered, so a holder's
remainder fare in that cache would be served to everyone; the cache, the floor audit and the
calibration sample all keep the operator's real full fare. Where the border sits comes from
the same feed-derived regions as everything else — Åsa is the last Halland stop, Kungsbacka
belongs to Västtrafik — and both price the tail at 90 SEK, so the conservative reading costs
the traveller nothing. If no fare exists for the remainder, or it is no cheaper than the
through fare (short-hop minimums do that), the single ticket stands.

Each card's region is the set of stops served by its home agency, extracted once from the
*full* GTFS feed (all route types, `data/tt-cache/agency-stops-*.json`) — county PTAs whose
only routes are local buses are absent from the intercity routing network, so their regions
must come from the whole feed. All 22 providers are supported, including the cross-region
passes Movingo and Norrlandsresan (modelled as the union of their constituent PTAs' regions).
Routing is pass-aware: when a card is held, the router's price floor is zeroed for the
operators it honors (zero is a valid lower bound), so a genuinely-free covered itinerary — a
multi-leg Kristianstad→Lund→Ystad trip inside Skåne — is not pruned before it can be priced.

**Mode selection.** The web UI's "Travel by" row and the CLI's `--modes train,bus,ferry,
freerider,flight` (and the `modes=` API param) choose which long-distance modes a search may
use; walk and local-transit feeders are always allowed. Journeys are combinations of the
ticked modes — tick train+car and you get train→free-car→train trips. Every ticked mode is
also guaranteed a representative in the results: if buses are cheapest, the list no longer
collapses to all buses; the cheapest train (and car, and flight) are pulled in at their
correct price position. Excluding a mode also skips its work — unchecking Flight avoids the
slow Google scrape, unchecking Freerider skips the inventory fetch.

**Round-trip and fare calendar.** `search --return DATE` prices there and back on their own
dates and sums the cheapest of each; `tripps fares` (and the "Cheapest day" button, and
`/api/fares`) prices every day over a window and shows which is cheapest, because these fares
are yield-managed — one Stockholm→Göteborg week ranged 335–585 SEK, so *when* to travel often
beats *how*. In the web UI each day is tappable, expanding to that day's full journey — every
leg with its operator, price, and a booking deeplink; `/api/fares` carries the full itinerary
per day so the expansion needs no second request. Each day is a full search; the built timetable is cached to disk (~2 MB, loads
in under a second versus ~15 s to parse), so the first window run is slow and the rest fast.

`tripps watch` turns the perishable Freerider inventory into a standing interest. Free cars
are first-come and vanish in minutes, so a date-specific search only finds one by luck.
Register a route and the background poller (or `tripps watch poll`) matches new cars against
it geographically — pickup within a radius of the origin, dropoff within a radius of the
destination, since depots sit outside town — and announces each once (a webhook if
configured, always the hit log). Every search also surfaces free cars on the route available
on *other* nearby dates, so the killer feature shows up even when you didn't search its date.
API: `POST/GET/DELETE /api/watch`.

`tripps calibrate` closes the price-floor feedback loop. The router optimizes a price *lower
bound* (`floors.py`), shipped with deliberately loose defaults; every phase-2 price is logged
to `reprice_delta`, and this command fits a tighter per-operator floor from that history —
`per_km = min(actual/distance)`, `base = min(actual - per_km·distance)`, both provably at or
below every observed fare, then discounted 25% (`SAFETY_MARGIN = 0.75`) for headroom against a
future fare cheaper than anything yet seen. Tighter floors mean a smaller Pareto frontier and
fewer upstream price calls. Run it periodically once real searches have accumulated; the
floor-violation detector catches and self-corrects any overshoot.

Endpoints: `GET /` (UI), `GET /status` (ops dashboard), `POST /search`, `GET /api/search`,
`GET /api/stops`, `GET /api/freerider`, `GET /health`, `GET /api/canary`.

## Deploy

```bash
docker compose up --build                  # builds the image, serves on :8000
```

No credentials, no env file: the entrypoint downloads the feed on first boot into the named
volume and starts serving. (Trafiklab documents a key on that download but does not currently
require one; `TRIPPS_TRAFIKLAB_GTFS_KEY` is wired through the compose file and sent when set,
so a future change stays a one-line fix rather than a rebuild.)

The image installs the package **editable** (`pip install -e ".[flights]"`) on purpose:
`config.PROJECT_ROOT` is `parents[2]` of the package, so an editable install keeps it — and the
data dir — resolving to `/app`, whereas a plain `pip install .` would relocate the package into
site-packages and break that path. `primp` — the TLS-fingerprint-impersonating client the Tora
rail adapter needs — is a core dependency, so the base install already prices rail; the
`[flights]` extra adds `fast-flights` for the Google Flights scrape and nothing else.
`docker-entrypoint.sh` fetches the feed once into the
`tripps-data` volume if it is empty, then runs `tripps serve`. That volume also holds the parsed
timetable cache and the SQLite DB, so a restart does not re-download or re-parse the ~500 MB
feed. The healthcheck gives a **120 s** start period for the lifespan warmup (feed parse ~15 s +
Freerider fetch + SJ key). Everything is configurable through `TRIPPS_*` env vars.

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
  passes.py            travel-card coverage engine + PassAdapter (region-gated free legs)
  data/travelcards.json  22-provider registry (honored operators + region model)
  ingest/              gtfs, freerider, flights, airports (OurAirports-derived)
  pricing/             flixbus, sj, tora, freerider, flights, operators (link-out), orchestrator
  search.py            Planner: resolve, overlay, route, price, rank.
  api/                 FastAPI + a dependency-free web UI.
```

## Verification

```bash
.venv/bin/python -m pytest tests/ -q     # 301 tests
.venv/bin/ruff check src tests
```

Tests run offline: the FlixBus and Freerider fixtures are recorded live responses, so a
schema change upstream fails a contract test instead of quietly mispricing a leg.

## Known gaps

- **Remaining unpriced operators.** Most rail and coach is now priced (SJ directly;
  Öresundståg, Mälartåg, Skånetrafiken, Tåg i Bergslagen and Vy Bus4You via Tora; the
  Flygbussarna airport coaches via the `StaticFareAdapter` fare table in `data/fares.json`).
  What is left — Y-buss, Härjedalingen, Masexpressen, and pure local transit — either sells
  through a different backend or has no obtainable price source, so those legs are routed and
  shown with a booking link.
  - **Ybuss / minor Turnit-tenant coaches** are the clearest gap: `booking.ybuss.se` is a
    Turnit SPA behind a TLS-fingerprint WAF with no public fares API, and its coach fares are
    yield-managed (so no honest fixed table). Pricing them would need a browser capture of the
    booking XHR per tenant; until that exists they stay link-out. Vy Bus4You, the largest
    Turnit coach, is already priced via Tora.
- **Zone-combination fares for partial cross-border card travel are not modelled.** A regional
  card that crosses a county border usually pays for the extra *zones* traversed, but the
  national GTFS feed strips all zone data (no `zone_id`, no fare files), and there are ~15
  different PTA zone systems, none CC0. So tripps prices those cross-border legs at the full
  single fare (conservative) rather than a zone-combined discount. The genuinely *free* named
  cross-border extensions (UL, Dalatrafik, Länstrafiken Örebro, VL — verified against each
  operator's own site) are modelled as `border_stops` allow-lists in the card registry.
- Local-transit legs are scheduled but never priced, and say so.
- The road matrix falls back to a great-circle estimate unless `TRIPPS_OSRM_BASE` is set.
- Legal: unofficial endpoints and scraping are used here for personal, non-commercial use.
  Systematic extraction and redisplay would need a look at each source's terms and at Swedish
  *katalogskydd* / the EU database right first.

## License

Not open source. You may read the code and run it for your own personal, non-commercial
travel planning; redistribution, commercial use, and running it as a service need written
permission. The reasoning is in [LICENSE](LICENSE), and it is the same reasoning as the
legal note above.
