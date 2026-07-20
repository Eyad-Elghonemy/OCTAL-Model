"""
usage_tracker.py

Operation logger backed by Supabase (a managed external database),
so log entries survive container restarts/redeploys -- unlike a local
JSON file, which lives on the container's ephemeral disk and is wiped
whenever the container is rebuilt.

Public interface (log_operation, get_logs) is unchanged from the
previous file-based version, so nothing else in the app needs to change.
"""

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from supabase import create_client

CAIRO_TZ = ZoneInfo("Africa/Cairo")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_TABLE_NAME = "operation_log"


def log_operation(operation_type: str, success: bool) -> None:
    """Record one API operation (e.g. 'linear', 'from-vin')."""
    supabase.table(_TABLE_NAME).insert({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation_type": operation_type,
        "success": success,
    }).execute()


def get_logs() -> list:
    """
    Returns all logged operations, most recent first, with the
    timestamp reformatted into a clean, display-ready string
    (e.g. '19/07/2026 11:16 PM') instead of raw ISO-8601 --
    identical output shape to the previous file-based version.
    """
    response = (
        supabase.table(_TABLE_NAME)
        .select("timestamp, operation_type, success")
        .order("id", desc=True)
        .execute()
    )

    formatted = []
    for e in response.data:
        try:
            dt = datetime.fromisoformat(e["timestamp"]).astimezone(CAIRO_TZ)
            display_time = dt.strftime("%d/%m/%Y %I:%M %p")
        except (KeyError, ValueError, TypeError):
            display_time = e.get("timestamp", "")
        formatted.append({
            "timestamp": display_time,
            "operation_type": e.get("operation_type"),
            "success": e.get("success"),
        })
    return formatted