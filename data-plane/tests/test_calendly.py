"""
End-to-end Calendly API smoke test.

Steps:
  1. GET  /users/me                    — resolve user + org URIs
  2. GET  /event_types?user=…          — list the account's event types
  3. GET  /event_type_available_times  — find open slots for the first active type
  4. POST /scheduling/invitees         — book the first available slot

Run with:
    CALENDLY_API_TOKEN=<your_pat> PYTHONPATH=. .venv/bin/python tests/test_calendly.py

Optional env overrides:
    CALENDLY_TIMEZONE=America/New_York   (default: UTC)
    CALENDLY_LOOKAHEAD_DAYS=14           (default: 30)
    CALENDLY_EVENT_TYPE_URI=https://api.calendly.com/event_types/...
                                         (skip discovery, use a specific event type)
"""
import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import httpx

# ── Config from environment ───────────────────────────────────────────────────

TOKEN = (
    os.environ.get("CALENDLY_API_TOKEN")
    or os.environ.get("CALENDLY_API_KEY")
    or ""
).strip()
if not TOKEN:
    print(
        "\n[ERROR] Neither CALENDLY_API_TOKEN nor CALENDLY_API_KEY is set.\n"
        "Set one and re-run:\n\n"
        "  export CALENDLY_API_KEY=<your_token>\n"
        "  PYTHONPATH=. .venv/bin/python tests/test_calendly.py\n"
    )
    sys.exit(1)

TZ             = os.environ.get("CALENDLY_TIMEZONE", "UTC")
LOOKAHEAD_DAYS = int(os.environ.get("CALENDLY_LOOKAHEAD_DAYS", "30"))
FORCED_ET_URI  = os.environ.get("CALENDLY_EVENT_TYPE_URI", "").strip() or None

BASE     = "https://api.calendly.com"
HEADERS  = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
INVITEE  = {
    "name":       "Test User",
    "first_name": "Test",
    "last_name":  "User",
    "email":      "test@example.com",
    "timezone":   TZ,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get(path: str, **params) -> dict:
    url = f"{BASE}{path}"
    r = httpx.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict) -> dict:
    url = f"{BASE}{path}"
    r = httpx.post(url, headers=HEADERS, json=body, timeout=15)
    r.raise_for_status()
    return r.json()


def pretty(obj: dict) -> str:
    return json.dumps(obj, indent=2)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


# ── Step 1: Resolve user URI ──────────────────────────────────────────────────
# The token may lack `users:read` scope, so we decode the UUID from the JWT
# payload directly rather than calling /users/me.

section("Step 1 — Resolve user URI from token")

def _jwt_payload(token: str) -> dict:
    """Decode the JWT payload without signature verification."""
    payload_b64 = token.split(".")[1]
    # Add padding if needed
    payload_b64 += "=" * (-len(payload_b64) % 4)
    return json.loads(base64.urlsafe_b64decode(payload_b64))

jwt_claims = _jwt_payload(TOKEN)
print(f"  Token scopes : {jwt_claims.get('scope', '(none)')}")
user_uuid = jwt_claims.get("user_uuid", "")
if not user_uuid:
    print("\n[ERROR] Could not find user_uuid in JWT — token may be malformed.")
    sys.exit(1)

user_uri = f"https://api.calendly.com/users/{user_uuid}"
print(f"  user_uri     : {user_uri}")

# Try /users/me to get org URI and timezone; gracefully skip if scope is missing
org_uri = None
try:
    me = get("/users/me")
    org_uri = me["resource"]["current_organization"]
    tz_from_profile = me["resource"].get("timezone", TZ)
    if TZ == "UTC" and tz_from_profile:
        TZ = tz_from_profile
        INVITEE["timezone"] = TZ
    print(f"  org_uri      : {org_uri}")
    print(f"  timezone     : {TZ}")
except httpx.HTTPStatusError as e:
    print(f"  /users/me returned {e.response.status_code} (scope likely missing) — using user URI only")
    print(f"  timezone     : {TZ} (env / default)")

# ── Step 2: List event types ──────────────────────────────────────────────────

section("Step 2 — GET /event_types")
et_data = get("/event_types", user=user_uri, count=50)
event_types = et_data.get("collection", [])
print(f"  Found {len(event_types)} event type(s):")
for et in event_types:
    active = et.get("active", False)
    print(f"    {'✓' if active else '✗'} [{et['kind']:6}] {et['name']:30}  {et['uri']}")

active_types = [et for et in event_types if et.get("active")]
if not active_types:
    print("\n[ERROR] No active event types found on this account.")
    sys.exit(1)

# Use forced URI if provided, otherwise pick the first active type
if FORCED_ET_URI:
    chosen_et = next((et for et in active_types if et["uri"] == FORCED_ET_URI), None)
    if not chosen_et:
        print(f"\n[ERROR] CALENDLY_EVENT_TYPE_URI not found among active types: {FORCED_ET_URI}")
        sys.exit(1)
else:
    chosen_et = active_types[0]

print(f"\n  Using event type: {chosen_et['name']}")
print(f"  URI             : {chosen_et['uri']}")
print(f"  Duration        : {chosen_et.get('duration', '?')} min")
print(f"  Scheduling URL  : {chosen_et.get('scheduling_url', '?')}")

# Fetch full event type details to discover location configuration
et_uuid = chosen_et["uri"].rsplit("/", 1)[-1]
et_detail = get(f"/event_types/{et_uuid}")["resource"]
locations = et_detail.get("locations", [])
print(f"  Locations       : {[l.get('kind') for l in locations]}")

# ── Step 3: Find available times ──────────────────────────────────────────────

section("Step 3 — GET /event_type_available_times")

# Calendly enforces a max 7-day window per call; page through weeks
now         = datetime.now(timezone.utc)
hard_end    = now + timedelta(days=LOOKAHEAD_DAYS + 1)
window_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
WINDOW_DAYS  = 7
MAX_SLOTS    = 5

available_slots = []
windows_checked = 0

while window_start < hard_end and len(available_slots) < MAX_SLOTS:
    window_end = min(window_start + timedelta(days=WINDOW_DAYS), hard_end)
    params = {
        "event_type": chosen_et["uri"],
        "start_time": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time":   window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    print(f"  Querying window {params['start_time']} → {params['end_time']} ...", end=" ", flush=True)

    avail_data = get("/event_type_available_times", **params)
    windows_checked += 1
    collection = avail_data.get("collection", [])
    open_slots = [
        s for s in collection
        if s.get("status") == "available" and s.get("invitees_remaining", 0) > 0
    ]
    print(f"{len(open_slots)} open slot(s) in this window")
    available_slots.extend(open_slots)
    window_start = window_end

if not available_slots:
    print(
        f"\n[WARN] No available slots found within {LOOKAHEAD_DAYS} days for "
        f"'{chosen_et['name']}'.\n"
        "       The test cannot proceed to booking — check the event type's availability schedule."
    )
    sys.exit(0)

available_slots = available_slots[:MAX_SLOTS]
print(f"\n  Found {len(available_slots)} available slot(s) (searched {windows_checked} window(s)):")
for i, s in enumerate(available_slots):
    print(f"    {i + 1}. {s['start_time']}  (remaining: {s.get('invitees_remaining', '?')})")

chosen_slot = available_slots[0]
print(f"\n  Booking slot: {chosen_slot['start_time']}")

# ── Step 4: Create booking via POST /scheduling/invitees ──────────────────────

section("Step 4 — POST /scheduling/invitees")

# Build location field from the event type's configured location kind.
# The API requires this to match exactly what the event type has set up.
booking_location = None
if locations:
    loc = locations[0]
    kind = loc.get("kind")
    booking_location = {"kind": kind}
    # Some kinds need extra fields
    if kind in ("physical", "custom"):
        booking_location["location"] = loc.get("location", "")
    elif kind == "outbound_call":
        booking_location["location"] = "+1 888-888-8888"   # placeholder
    print(f"  Using location  : {booking_location}")
else:
    print("  No location configured on event type — omitting location field")

booking_body: dict = {
    "event_type":     chosen_et["uri"],
    "start_time":     chosen_slot["start_time"],
    "invitee":        INVITEE,
    "booking_source": "ai_scheduling_assistant",
}
if booking_location:
    booking_body["location"] = booking_location

print("  Request body:")
print("  " + pretty(booking_body).replace("\n", "\n  "))

# Calendly's scheduling API endpoint for AI/programmatic booking.
# Falls back through candidate paths so we can identify the correct one.
_BOOKING_ENDPOINTS = ["/invitees"]

booking_result = None
for endpoint in _BOOKING_ENDPOINTS:
    print(f"  Trying POST {endpoint} ...")
    try:
        booking_result = post(endpoint, booking_body)
        print(f"  → success on {endpoint}")
        break
    except httpx.HTTPStatusError as exc:
        body_text = ""
        try:
            body_text = pretty(exc.response.json())
        except Exception:
            body_text = exc.response.text[:300]
        print(f"  → HTTP {exc.response.status_code}: {body_text}")
        if exc.response.status_code not in (404, 405):
            # A 4xx other than "not found / method not allowed" is a real error
            section("✗  Booking failed")
            sys.exit(1)

if booking_result is None:
    section("✗  Booking failed — no endpoint accepted the request")
    print("  The Calendly scheduling API may require a paid plan or a different scope.")
    sys.exit(1)

resource = booking_result.get("resource", {})
section("✓  Booking confirmed!")
print(f"  invitee_uri    : {resource.get('uri', '?')}")
print(f"  status         : {resource.get('status', '?')}")
print(f"  event_uri      : {resource.get('event', '?')}")
print(f"  cancel_url     : {resource.get('cancel_url', '?')}")
print(f"  reschedule_url : {resource.get('reschedule_url', '?')}")
print(f"  created_at     : {resource.get('created_at', '?')}")
print()
