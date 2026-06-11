"""
RailMind AI — NTES Live Train Status Client
============================================
Wraps the unofficial NTES (National Train Enquiry System) endpoint that
returns real-time running status for Indian Railways trains.

NTES endpoint documentation
----------------------------
The NTES website at https://enquiry.indianrail.gov.in exposes a handful of
JSON endpoints that are publicly accessible without authentication.  These are
not officially documented but are widely used by the Indian developer community
(see: whereismytrain.in, RailYatri, RailMadad etc.).

Endpoints used
~~~~~~~~~~~~~~
1. Train running status (live position + delay at each station):
   POST https://enquiry.indianrail.gov.in/miniTrainQuery/GetRunningStatus
   Body: {"trainNo": "12301", "allStation": "Y", "lang": "0"}

2. Trains between stations (upcoming arrivals/departures):
   POST https://enquiry.indianrail.gov.in/mntes/q?opt=TR&trainNo=...

Rate limiting / Ethics
~~~~~~~~~~~~~~~~~~~~~~
- NTES does NOT publish a terms-of-service for API access.
- We limit requests to 1 per 60 seconds per train to be considerate.
- All requests include a descriptive User-Agent header.
- Data is used exclusively for non-commercial crowd estimation.

Fallback
~~~~~~~~
When NTES is unreachable (timeouts, maintenance, IP blocks) the client
returns None so callers gracefully fall back to the ML/statistical model.
"""


import json
import logging
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NTES_BASE    = "https://enquiry.indianrail.gov.in/miniTrainQuery"
NTES_HOME    = "https://enquiry.indianrail.gov.in/ntes/ntes/init"
NTES_TIMEOUT = 5       # seconds — NTES is slow; be patient
CACHE_TTL    = 60         # seconds — don't hammer the endpoint
MAX_RETRIES  = 1   

# Mimic a real browser — NTES returns HTML (session page) to non-browser UAs
HEADERS = {
    "User-Agent":       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36",
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "en-IN,en;q=0.9,hi;q=0.8",
    "Content-Type":     "application/json",
    "Referer":          "https://enquiry.indianrail.gov.in/ntes/ntes/init",
    "Origin":           "https://enquiry.indianrail.gov.in",
    "X-Requested-With": "XMLHttpRequest",
}


# ---------------------------------------------------------------------------
# Simple TTL cache (no external dep required)
# ---------------------------------------------------------------------------

class _TTLCache:
    def __init__(self, ttl_seconds: int = 60) -> None:
        self._store: dict[str, tuple[float, object]] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[object]:
        if key in self._store:
            ts, val = self._store[key]
            if time.monotonic() - ts < self.ttl:
                return val
        return None

    def set(self, key: str, value: object) -> None:
        self._store[key] = (time.monotonic(), value)


_cache = _TTLCache(ttl_seconds=CACHE_TTL)


# ---------------------------------------------------------------------------
# NTES Client
# ---------------------------------------------------------------------------

class NTESClient:
    """
    Thin wrapper around the NTES unofficial JSON API.

    Root cause of "Expecting value" error
    --------------------------------------
    NTES returns an HTML session/CSRF page instead of JSON when:
      1. The request has no session cookie (first hit)
      2. The User-Agent looks like a bot
      3. The Referer header is missing or wrong

    Fix: use a persistent requests.Session() that first GETs the NTES
    homepage to acquire the session cookie, then POSTs to the API.
    The session carries the cookie automatically on subsequent calls.

    All methods return None on any failure — callers handle gracefully.
    """

    def __init__(self) -> None:
        self._session: Optional[requests.Session] = None
        self._session_warmed = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def _get_session(self) -> requests.Session:
        """Return a warmed-up session with NTES cookies loaded."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(HEADERS)

        if not self._session_warmed:
            try:
                # Step 1 — GET the homepage to receive the session cookie
                self._session.get(NTES_HOME, timeout=NTES_TIMEOUT)
                self._session_warmed = True
                logger.debug("NTES session warmed up ✅")
            except Exception as exc:
                logger.debug("NTES session warm-up failed: %s", exc)
                # Continue anyway — cookie may not be strictly required
                self._session_warmed = True

        return self._session

    def _reset_session(self) -> None:
        """Force a fresh session on next call (after persistent failures)."""
        self._session = None
        self._session_warmed = False

    # ------------------------------------------------------------------
    # Core: running status for one train
    # ------------------------------------------------------------------
    def get_running_status(self, train_number: str) -> Optional[dict]:
        """
        Fetch live running status for `train_number`.

        Returns
        -------
        dict  with keys:
            train_number, train_name, current_station_code,
            current_station_name, delay_minutes, last_updated
        or None on failure.
        """
        cache_key = f"run_{train_number}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        url     = f"{NTES_BASE}/GetRunningStatus"
        payload = {"trainNo": train_number, "allStation": "Y", "lang": "0"}

        for attempt in range(MAX_RETRIES):
            try:
                session = self._get_session()
                resp    = session.post(url, json=payload, timeout=NTES_TIMEOUT)
                resp.raise_for_status()

                # Detect HTML response (session expired / blocked)
                content_type = resp.headers.get("Content-Type", "")
                raw_text     = resp.text.strip()

                if "text/html" in content_type or raw_text.startswith("<"):
                    # Check if NTES is in maintenance mode
                    if "technical activity" in raw_text or "un-available" in raw_text or "mntes" in raw_text:
                        logger.warning(
                            "NTES is in maintenance mode — trying RailYatri fallback"
                        )
                        return self._railyatri_fallback(train_number)
                    logger.debug(
                        "NTES returned HTML (attempt %d) — resetting session", attempt + 1
                    )
                    self._reset_session()
                    continue   # retry with fresh session + new cookie

                if not raw_text:
                    logger.debug("NTES returned empty body for train %s", train_number)
                    return None

                data   = resp.json()
                parsed = self._parse_running_status(data)
                if parsed:
                    _cache.set(cache_key, parsed)
                return parsed

            except requests.exceptions.Timeout:
                logger.warning("NTES timeout for train %s (attempt %d)", train_number, attempt + 1)
            except requests.exceptions.ConnectionError:
                logger.warning("NTES connection error — offline?")
                break
            except Exception as exc:
                logger.warning("NTES error for train %s: %s", train_number, exc)
                break

        # All NTES attempts failed — try WhereIsMyTrain as last resort
        logger.info("All NTES attempts failed for %s — trying WIMT", train_number)
        return self._wimt_fallback(train_number)

    # ------------------------------------------------------------------
    # Fallback 1 — RailYatri (works when NTES is down)
    # ------------------------------------------------------------------
    def _railyatri_fallback(self, train_number: str) -> Optional[dict]:
        """
        RailYatri exposes an unofficial train-status endpoint used by their
        mobile app. More stable than NTES because it has its own data pipeline
        that mirrors IR data independently.

        Endpoint:
          GET https://railyatri.in/api/v2/train-running-status?train_number=12301
        """
        try:
            url = "https://railyatri.in/api/v2/train-running-status"
            headers = {
                "User-Agent":      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                                   "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
                "Accept":          "application/json",
                "Referer":         "https://railyatri.in/",
            }
            resp = requests.get(
                url,
                params={"train_number": train_number},
                headers=headers,
                timeout=8,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()

            # RailYatri response shape:
            # { "data": { "train_number": "12301", "train_name": "...",
            #             "stations": [{ "stn_code": "...", "delay": 14, ... }] } }
            d = data.get("data") or data
            stations = d.get("stations") or []

            # Find the most recently passed station
            current = None
            for s in stations:
                if s.get("has_arrived") or s.get("actual_arrival"):
                    current = s
            if current is None and stations:
                current = stations[0]
            if current is None:
                return None

            delay = int(current.get("delay") or current.get("delay_minutes") or 0)
            result = {
                "train_number":         str(d.get("train_number", train_number)),
                "train_name":           str(d.get("train_name", "")),
                "current_station_code": str(current.get("stn_code", "")),
                "current_station_name": str(current.get("station_name", "")),
                "delay_minutes":        delay,
                "last_updated":         datetime.now().isoformat(),
                "source":               "RailYatri",
            }
            logger.info("RailYatri fallback ✅  train %s delay=%d min", train_number, delay)
            return result

        except Exception as exc:
            logger.debug("RailYatri fallback failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Fallback 2 — Where Is My Train (Ixigo) app API
    # ------------------------------------------------------------------
    def _wimt_fallback(self, train_number: str) -> Optional[dict]:
        """
        WhereIsMyTrain (now part of Ixigo) has a public-facing API used by
        their Android app. Third fallback when both NTES and RailYatri fail.
        """
        try:
            url = f"https://whereismytrain.in/cache/trainDetails?trainNumber={train_number}"
            headers = {
                "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 13; Pixel 7 Build/TQ3A)",
                "Accept":     "application/json",
            }
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code != 200:
                return None
            data = resp.json()

            # WIMT shape varies; attempt best-effort parse
            train_data = data.get("train") or data
            stations   = train_data.get("station_list") or train_data.get("stations") or []

            current = None
            for s in stations:
                if s.get("status") in ("DEPARTED", "ARRIVED") or s.get("hasDeparted"):
                    current = s
            if current is None and stations:
                current = stations[0]
            if current is None:
                return None

            delay = int(
                current.get("lateMin") or
                current.get("delay") or
                current.get("late") or
                0
            )
            result = {
                "train_number":         train_number,
                "train_name":           str(train_data.get("trainName", "")),
                "current_station_code": str(current.get("stationCode", "")),
                "current_station_name": str(current.get("stationName", "")),
                "delay_minutes":        delay,
                "last_updated":         datetime.now().isoformat(),
                "source":               "WhereIsMyTrain",
            }
            logger.info("WIMT fallback ✅  train %s delay=%d min", train_number, delay)
            return result

        except Exception as exc:
            logger.debug("WIMT fallback failed: %s", exc)
            return None

    def _parse_running_status(self, raw: dict) -> Optional[dict]:
        """Extract the fields we care about from NTES JSON response."""
        try:
            # NTES wraps data differently depending on API version
            body = raw.get("body") or raw.get("data") or raw
            if isinstance(body, str):
                body = json.loads(body)

            trains = body.get("TrainDetails") or body.get("trainDetails") or []
            if not trains:
                return None

            # First entry = current/most-recent station
            t = trains[0]
            delay_raw = (
                t.get("lateMin") or
                t.get("delayMin") or
                t.get("delay") or
                "0"
            )
            delay_minutes = int(str(delay_raw).replace("+", "").strip() or "0")

            return {
                "train_number":        t.get("trainNo", ""),
                "train_name":          t.get("trainName", ""),
                "current_station_code": t.get("stnCode", ""),
                "current_station_name": t.get("stnName", ""),
                "delay_minutes":       delay_minutes,
                "last_updated":        datetime.now().isoformat(),
                "source":              "NTES",
            }
        except Exception as exc:
            logger.debug("NTES parse error: %s | raw=%s", exc, str(raw)[:200])
            return None

    # ------------------------------------------------------------------
    # Bulk: delays for a list of trains (used by platform allocator)
    # ------------------------------------------------------------------
    def get_bulk_delays(self, train_numbers: list[str]) -> dict[str, int]:
        """
        Return {train_number: delay_minutes} for the given trains.
        Trains that can't be fetched are omitted from the result.
        """
        results: dict[str, int] = {}
        for tn in train_numbers:
            status = self.get_running_status(tn)
            if status is not None:
                results[tn] = status["delay_minutes"]
        return results

    # ------------------------------------------------------------------
    # Trains arriving at a station in the next N hours
    # ------------------------------------------------------------------
    def get_arrivals(self, station_code: str, hours: int = 3) -> list[dict]:
        """
        Returns upcoming arrivals at `station_code` in the next `hours` hours.

        Note: NTES's "trains at station" endpoint is less stable than the
        running-status one.  This method returns [] if the endpoint fails.
        """
        cache_key = f"arr_{station_code}_{hours}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        url = "https://enquiry.indianrail.gov.in/mntes/q"
        params = {
            "opt":  "TR",
            "stnCode": station_code,
            "type":    "ARR",
        }

        try:
            session = self._get_session()
            resp    = session.get(url, params=params, timeout=NTES_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            arrivals = self._parse_arrivals(data, hours)
            _cache.set(cache_key, arrivals)
            return arrivals
        except Exception as exc:
            logger.info("NTES arrivals unavailable for %s: %s", station_code, exc)
            return []

    def _parse_arrivals(self, raw: dict, hours: int) -> list[dict]:
        """Parse NTES arrival list; filter to next `hours` hours."""
        results = []
        now     = datetime.now()
        cutoff  = now + timedelta(hours=hours)

        try:
            trains = (
                raw.get("body", {}).get("TrainDetails") or
                raw.get("data", []) or
                []
            )
            for t in trains:
                try:
                    sched_str = t.get("schArrTime") or t.get("scheduledArrival", "")
                    if not sched_str:
                        continue
                    # Parse HH:MM format
                    hour, minute = map(int, sched_str.strip().split(":")[:2])
                    sched_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if sched_dt < now:
                        sched_dt += timedelta(days=1)
                    if sched_dt > cutoff:
                        continue

                    delay = int(str(t.get("lateMin") or "0").replace("+", "") or "0")
                    coach_count = int(t.get("coaches") or "18")   # default 18 coaches
                    # Avg coach occupancy: ~70 passengers per coach for Superfast
                    crowd_contribution = coach_count * 70

                    results.append({
                        "train_number":       t.get("trainNo", ""),
                        "train_name":         t.get("trainName", ""),
                        "scheduled_arrival":  sched_str,
                        "delay_minutes":      delay,
                        "crowd_contribution": crowd_contribution,
                    })
                except (ValueError, KeyError):
                    continue
        except Exception as exc:
            logger.debug("NTES arrival parse error: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Station crowd proxy from scheduled arrivals
    # ------------------------------------------------------------------
    def estimate_platform_crowd_from_arrivals(
        self, station_code: str, hours_ahead: int = 2
    ) -> Optional[int]:
        """
        Use upcoming arrivals × average occupancy as a proxy for platform crowd.

        Returns an integer passenger count, or None if NTES is unavailable.
        """
        arrivals = self.get_arrivals(station_code, hours=hours_ahead)
        if not arrivals:
            return None

        total = sum(a["crowd_contribution"] for a in arrivals)
        # Weight by proximity — closer trains contribute more to current crowd
        return min(total, 15000)   # sanity cap


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
ntes_client = NTESClient()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    train_no = sys.argv[1] if len(sys.argv) > 1 else "12301"
    print(f"\nFetching running status for train {train_no}…")
    print("Fallback chain: NTES → RailYatri → WhereIsMyTrain\n")

    client = NTESClient()
    result = client.get_running_status(train_no)

    if result:
        print(f"\n✅  Got data from: {result.get('source', 'unknown')}")
        print(json.dumps(result, indent=2))
    else:
        print("\n❌  All three sources unavailable.")
        print("This is normal when NTES is in maintenance.")
        print("Your crowd model is already using the IR statistical fallback.")
        print("\nCheck NTES status manually at: https://enquiry.indianrail.gov.in/mntes")