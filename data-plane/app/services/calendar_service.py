"""Calendly v2 calendar integration.

Correct API flow (aligned with calendly/buzzwordcrm reference implementation):

  1. GET  /event_type_available_times  — find open slots for a given event type
  2. POST /scheduling/invitees         — create a booking with full invitee details

NOT used (our earlier mistake):
  - GET  /scheduled_events             — lists already-booked events, not open slots
  - POST /scheduled_events/{uuid}/invitees — adds invitee to existing group event slots

Falls back to a dummy implementation when no Calendly config is present.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_CALENDLY_BASE = "https://api.calendly.com"
_MAX_SLOTS = 5      # present at most this many options over the phone
# Calendly requires the search window to be ≤ 7 days per request
_WINDOW_DAYS = 7


# ── TimeSlot dataclass ────────────────────────────────────────────────────────

@dataclass
class TimeSlot:
    slot_id: str            # unique id — ISO start_time used as stable key
    start: datetime
    end: datetime
    description: str        # voice-friendly label, e.g. "Monday, March 16 at 10:00 AM"
    event_type_uri: str = ""   # Calendly event type URI (needed for booking)
    spots_remaining: int = 1


# ── Calendly helpers ──────────────────────────────────────────────────────────

def _headers(api_token: str) -> dict:
    return {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}


def _slot_description(start: datetime, tz: str) -> str:
    """Format a UTC datetime as a voice-friendly local-time string."""
    try:
        from zoneinfo import ZoneInfo
        local = start.astimezone(ZoneInfo(tz))
    except Exception:
        local = start
    return local.strftime("%A, %B %-d at %-I:%M %p")


def _parse_dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _split_name(full_name: str) -> tuple[str, str]:
    """Split 'First Last' into (first, last). Handles single-word names."""
    parts = full_name.strip().split(maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")


def _user_uri_from_token(api_token: str) -> str | None:
    """
    Decode the Calendly PAT (JWT) to extract the user URI without an API call.
    Falls back to GET /users/me if the JWT decode fails.
    """
    import base64
    import json as _json
    try:
        payload_b64 = api_token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        claims = _json.loads(base64.urlsafe_b64decode(payload_b64))
        user_uuid = claims.get("user_uuid", "")
        if user_uuid:
            return f"https://api.calendly.com/users/{user_uuid}"
    except Exception:
        pass

    # Fallback: ask the API
    try:
        with httpx.Client(base_url=_CALENDLY_BASE, timeout=10.0) as client:
            resp = client.get("/users/me", headers=_headers(api_token))
            resp.raise_for_status()
            return resp.json()["resource"]["uri"]
    except Exception as exc:
        logger.warning("Could not resolve Calendly user URI: %s", exc)
        return None


def _discover_event_type_uri(api_token: str, scheduling_link: str | None = None) -> str | None:
    """
    Resolve a Calendly event type URI.

    Priority:
      1. Match scheduling_link against event type scheduling_url fields
         (e.g. CALENDLY_SCHEDULING_LINK=https://calendly.com/user/event-slug)
      2. Fall back to the first active event type on the account
    """
    user_uri = _user_uri_from_token(api_token)
    if not user_uri:
        return None

    try:
        with httpx.Client(base_url=_CALENDLY_BASE, timeout=10.0) as client:
            resp = client.get(
                "/event_types",
                headers=_headers(api_token),
                params={"user": user_uri, "count": 50},
            )
            resp.raise_for_status()
        event_types = resp.json().get("collection", [])
        active = [et for et in event_types if et.get("active")]

        # Priority 1: match by scheduling link
        if scheduling_link:
            # Normalize: strip trailing slash for comparison
            link = scheduling_link.rstrip("/")
            for et in active:
                if et.get("scheduling_url", "").rstrip("/") == link:
                    logger.info(
                        "Calendly: resolved event_type from scheduling link | name=%r uri=%s",
                        et.get("name"), et["uri"],
                    )
                    return et["uri"]
            logger.warning(
                "Calendly: scheduling_link %r did not match any active event type — falling back to first",
                scheduling_link,
            )

        # Priority 2: first active event type
        if active:
            uri = active[0]["uri"]
            logger.info(
                "Calendly auto-discovery: using first active event_type=%r (%s)",
                active[0].get("name"), uri,
            )
            return uri

        logger.warning("Calendly auto-discovery: no active event types found")
    except Exception as exc:
        logger.warning("Calendly event type discovery failed: %s", exc)
    return None


# ── Calendly: list available times ───────────────────────────────────────────
# Mirrors CalendlyService.getUserEventTypeAvailTimes from the reference impl.
# GET /event_type_available_times?event_type=&start_time=&end_time=

def _get_event_type_location(api_token: str, event_type_uri: str) -> dict | None:
    """
    Fetch the first configured location kind for an event type.

    The POST /invitees API requires a `location` field whose `kind` matches
    what the event type has configured — omitting it or using the wrong kind
    returns a 400 "invalid_location_choice" error.
    """
    uuid = event_type_uri.rsplit("/", 1)[-1]
    try:
        with httpx.Client(base_url=_CALENDLY_BASE, timeout=10.0) as client:
            resp = client.get(f"/event_types/{uuid}", headers=_headers(api_token))
            resp.raise_for_status()
        locations = resp.json().get("resource", {}).get("locations", [])
        if not locations:
            return None
        loc = locations[0]
        kind = loc.get("kind")
        result: dict = {"kind": kind}
        if kind in ("physical", "custom"):
            result["location"] = loc.get("location", "")
        return result
    except Exception as exc:
        logger.warning("Could not fetch event type location: %s", exc)
        return None


def _list_calendly_slots(
    calendly_cfg: dict,
    event_type_uri: str,
    search_start: Optional[datetime] = None,
    search_end: Optional[datetime] = None,
) -> list[TimeSlot]:
    """
    Fetch open booking slots via GET /event_type_available_times.

    Calendly limits the query window to 7 days per request, so we page through
    7-day windows starting from tomorrow (or search_start) until we have
    _MAX_SLOTS results or we exceed the search boundary.

    search_start / search_end override the default lookahead window and are
    used for user-requested date ranges (e.g. "next Tuesday morning").
    """
    api_token = calendly_cfg["api_token"]
    lookahead_days = int(calendly_cfg.get("lookahead_days", 30))
    tz = calendly_cfg.get("timezone", "UTC")

    now = datetime.now(timezone.utc)
    if search_start is not None:
        window_start = search_start
        hard_end = search_end if search_end is not None else search_start + timedelta(days=7)
    else:
        window_start = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        hard_end = now + timedelta(days=lookahead_days + 1)

    slots: list[TimeSlot] = []

    logger.info(
        "Calendly: fetching available times | event_type_uri=%s lookahead_days=%d",
        event_type_uri, lookahead_days,
    )

    try:
        with httpx.Client(base_url=_CALENDLY_BASE, timeout=10.0) as client:
            while window_start < hard_end and len(slots) < _MAX_SLOTS:
                window_end = min(
                    window_start + timedelta(days=_WINDOW_DAYS),
                    hard_end,
                )

                params = {
                    "event_type": event_type_uri,
                    "start_time": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_time": window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }

                resp = client.get(
                    "/event_type_available_times",
                    headers=_headers(api_token),
                    params=params,
                )
                resp.raise_for_status()

                for item in resp.json().get("collection", []):
                    if item.get("status") != "available":
                        continue
                    if item.get("invitees_remaining", 0) < 1:
                        continue

                    start_dt = _parse_dt(item["start_time"])
                    # Calendly available_times doesn't return end_time; we leave it
                    # as start + 1h by convention (the LLM only reads description).
                    end_dt = start_dt + timedelta(hours=1)

                    slots.append(TimeSlot(
                        slot_id=item["start_time"],   # ISO string as stable key
                        start=start_dt,
                        end=end_dt,
                        description=_slot_description(start_dt, tz),
                        event_type_uri=event_type_uri,
                        spots_remaining=item.get("invitees_remaining", 1),
                    ))

                    if len(slots) >= _MAX_SLOTS:
                        break

                window_start = window_end

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Calendly available_times HTTP %s: %s",
            exc.response.status_code, exc.response.text[:300],
        )
    except Exception as exc:
        logger.error("Calendly available_times error: %s", exc)

    logger.info(
        "Calendly: found %d available slot(s) for event_type_uri=%s",
        len(slots), event_type_uri,
    )
    return slots


# ── Calendly: create booking via scheduling API ───────────────────────────────
# POST /scheduling/invitees — creates a new event + invitee in one call.
# Request body matches the format in the user spec (booking_source, first_name,
# last_name, text_reminder_number, tracking, questions_and_answers, etc.).

def _book_calendly_slot(
    calendly_cfg: dict,
    slot: TimeSlot,
    collected: dict,
) -> dict:
    """
    Create a Calendly booking via POST /scheduling/invitees.

    Extracts name, email, phone from collected params and builds the full
    request body including booking_source = "ai_scheduling_assistant".
    """
    api_token = calendly_cfg["api_token"]
    tz = calendly_cfg.get("timezone", "UTC")

    logger.info("Calendly: collected keys at booking time: %s", list(collected.keys()))
    full_name = _extract_name(collected)
    first_name, last_name = _split_name(full_name)
    email = _extract_email(collected)
    phone = _extract_phone(collected)

    if not email:
        logger.warning("No email in collected params — Calendly booking may fail")

    # Calendly requires E.164 format (+12025551234).
    # Normalize 10-digit US numbers and strip spaces/dashes/parens.
    if phone:
        digits = re.sub(r"\D", "", phone)
        if len(digits) == 10:
            phone = f"+1{digits}"
        elif len(digits) == 11 and digits.startswith("1"):
            phone = f"+{digits}"
        elif not phone.startswith("+"):
            phone = f"+{digits}"

    invitee: dict = {
        "name": full_name,
        "first_name": first_name,
        "email": email,
        "timezone": tz,
    }
    if last_name:
        invitee["last_name"] = last_name
    if phone:
        invitee["text_reminder_number"] = phone

    body: dict = {
        "event_type": slot.event_type_uri,
        "start_time": slot.start.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
        "invitee": invitee,
        "booking_source": "ai_scheduling_assistant",
    }

    # The API rejects requests where `location.kind` doesn't match the event
    # type's configured location — fetch it and include it in the body.
    location = _get_event_type_location(api_token, slot.event_type_uri)
    if location:
        body["location"] = location

    import json as _json
    logger.info(
        "Calendly: scheduling invitee | event_type_uri=%s start=%s name=%r email=%r location_kind=%s",
        slot.event_type_uri, body["start_time"], full_name, email,
        location.get("kind") if location else "none",
    )
    logger.info("Calendly: POST /invitees request body: %s", _json.dumps(body))

    with httpx.Client(base_url=_CALENDLY_BASE, timeout=10.0) as client:
        resp = client.post(
            "/invitees",
            headers=_headers(api_token),
            json=body,
        )
        if not resp.is_success:
            logger.error(
                "Calendly: POST /invitees failed | status=%d body=%s",
                resp.status_code, resp.text[:1000],
            )
        resp.raise_for_status()

    resource = resp.json().get("resource", {})
    invitee_uri: str = resource.get("uri", "")
    invitee_uuid = invitee_uri.rsplit("/", 1)[-1] if invitee_uri else "unknown"

    logger.info("Calendly: booking confirmed | invitee_uuid=%s", invitee_uuid)
    return {
        "booking_id": invitee_uuid,
        "invitee_uri": invitee_uri,
        "status": resource.get("status", "active"),
        "event_uri": resource.get("event", ""),
        "cancel_url": resource.get("cancel_url", ""),
        "reschedule_url": resource.get("reschedule_url", ""),
        "confirmed_at": resource.get("created_at", datetime.now(timezone.utc).isoformat()),
    }


# ── Dummy fallback ────────────────────────────────────────────────────────────

def _list_dummy_slots(purpose: str) -> list[TimeSlot]:
    """Generate synthetic slots for the next three weekdays (no Calendly account needed)."""
    logger.info("Calendar (dummy): listing slots for purpose=%r", purpose)
    now = datetime.now(timezone.utc)
    slots: list[TimeSlot] = []

    days_added = 0
    offset = 1
    while days_added < 3:
        candidate = now + timedelta(days=offset)
        offset += 1
        if candidate.weekday() >= 5:
            continue
        day_label = candidate.strftime("%A, %B %-d")
        for hour, label in ((10, "10:00 AM"), (14, "2:00 PM")):
            start = candidate.replace(hour=hour, minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
            slots.append(TimeSlot(
                slot_id=f"dummy-{days_added + 1}-{'am' if hour == 10 else 'pm'}",
                start=start,
                end=end,
                description=f"{day_label} at {label}",
            ))
        days_added += 1

    return slots


def _book_dummy_slot(slot: TimeSlot, params: dict) -> dict:
    logger.info("Calendar (dummy): booking slot_id=%r", slot.slot_id)
    booking_id = f"BK-{slot.slot_id.upper()}-{int(datetime.now(timezone.utc).timestamp())}"
    return {
        "booking_id": booking_id,
        "slot_id": slot.slot_id,
        "status": "confirmed",
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Collected-params extraction helpers ──────────────────────────────────────

def _extract_name(collected: dict) -> str:
    for key in (
        "client_name", "client_full_name", "caller_name",
        "full_name", "name", "contact_name", "guest_name",
    ):
        if collected.get(key):
            return collected[key]
    # Last resort: any key whose name suggests it holds a person's name
    for key, val in collected.items():
        if val and "name" in key.lower():
            return val
    return "Unknown Caller"


def _extract_email(collected: dict) -> str:
    for key in ("email", "email_address", "client_email", "contact_email"):
        if collected.get(key):
            return collected[key]
    return ""


def _extract_phone(collected: dict) -> str:
    for key in (
        "phone", "phone_number", "mobile", "cell",
        "text_reminder_number", "contact_phone", "client_phone",
    ):
        if collected.get(key):
            return collected[key]
    return ""


# ── Public interface ──────────────────────────────────────────────────────────

def _resolve_calendly_cfg(config: dict) -> dict | None:
    """
    Return a usable Calendly config dict, or None if Calendly is not configured.

    Priority:
      1. Control-plane export  (config["calendly"] — set via admin API)
      2. Data-plane .env       (CALENDLY_API_KEY — useful for local dev/testing)
    """
    cfg = config.get("calendly")
    if cfg and cfg.get("api_token"):
        return cfg

    # Fallback: read directly from data-plane settings
    try:
        from app.config import settings
        if settings.calendly_api_key:
            return {
                "api_token": settings.calendly_api_key,
                "scheduling_link": settings.calendly_scheduling_link or None,
                "user_uri": None,
                "organization_uri": None,
                "lookahead_days": 30,
                "timezone": settings.calendly_timezone,
            }
    except Exception:
        pass

    return None


def list_available_slots(
    purpose: str,
    config: dict,
    event_type_uri: Optional[str] = None,
    search_start: Optional[datetime] = None,
    search_end: Optional[datetime] = None,
) -> list[TimeSlot]:
    """
    Return available appointment slots for the given purpose.

    Uses Calendly GET /event_type_available_times when a config and
    event_type_uri are provided; falls back to dummy slots otherwise.

    search_start / search_end constrain the search window (for user-specified
    date ranges such as "next Tuesday morning").
    """
    calendly_cfg = _resolve_calendly_cfg(config)
    if not calendly_cfg:
        return _list_dummy_slots(purpose)

    # Auto-discover the event type when none is explicitly configured.
    # Also resolve scheduling URLs (https://calendly.com/...) to API URIs
    # (https://api.calendly.com/event_types/{uuid}) — the available_times
    # endpoint only accepts the API URI form.
    raw_uri = event_type_uri or calendly_cfg.get("scheduling_link")
    if raw_uri and not raw_uri.startswith("https://api.calendly.com/"):
        # It's a scheduling URL — look it up via the event types list
        resolved_uri = _discover_event_type_uri(calendly_cfg["api_token"], raw_uri)
        if not resolved_uri:
            logger.warning(
                "Could not resolve scheduling URL %r to an API URI — falling back to first event type",
                raw_uri,
            )
            resolved_uri = _discover_event_type_uri(calendly_cfg["api_token"], None)
    else:
        resolved_uri = raw_uri or _discover_event_type_uri(
            calendly_cfg["api_token"], calendly_cfg.get("scheduling_link")
        )

    if resolved_uri:
        return _list_calendly_slots(calendly_cfg, resolved_uri, search_start, search_end)

    logger.warning("Calendly config present but no event type URI available — using dummy slots")
    return _list_dummy_slots(purpose)


def book_time_slot(slot: TimeSlot, collected: dict, config: dict) -> dict:
    """
    Book the given slot for the caller.

    Uses Calendly POST /invitees when a Calendly config is present
    and the slot carries an event_type_uri; falls back to dummy booking.
    """
    calendly_cfg = _resolve_calendly_cfg(config)
    if not calendly_cfg:
        return _book_dummy_slot(slot, collected)

    # Ensure the slot carries an API URI, not a scheduling URL.
    if not slot.event_type_uri or not slot.event_type_uri.startswith("https://api.calendly.com/"):
        scheduling_link = slot.event_type_uri if slot.event_type_uri else calendly_cfg.get("scheduling_link")
        discovered = _discover_event_type_uri(calendly_cfg["api_token"], scheduling_link)
        if discovered:
            slot.event_type_uri = discovered

    if slot.event_type_uri:
        return _book_calendly_slot(calendly_cfg, slot, collected)

    logger.warning("book_time_slot: no event_type_uri on slot — falling back to dummy booking")
    return _book_dummy_slot(slot, collected)
