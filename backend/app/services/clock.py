"""
INJECTABLE CLINICAL CLOCK & TIMEZONE SERVICE (clock.py)
-------------------------------------------------------
Provides deterministic, timezone-aware date and time resolution for Viscan AI.
Eliminates hardcoded date strings, supports injectable clocks for testing,
and dynamically formats relative dates ("Today", "Tomorrow", calendar dates).
"""

from datetime import datetime, date, timedelta
from typing import Optional, Union
import zoneinfo

# Default clinical workstation timezone
DEFAULT_TIMEZONE = "Asia/Kolkata"

# Global frozen datetime for deterministic test execution
_FROZEN_DATETIME: Optional[datetime] = None


def set_frozen_clock(dt: Optional[Union[datetime, str]] = None):
    """
    Sets a global frozen datetime for deterministic testing.
    Pass None to unfreeze and restore live system clock.
    """
    global _FROZEN_DATETIME
    if dt is None:
        _FROZEN_DATETIME = None
    elif isinstance(dt, str):
        # Support ISO formats 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        if " " in dt:
            _FROZEN_DATETIME = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        elif "T" in dt:
            _FROZEN_DATETIME = datetime.fromisoformat(dt)
        else:
            _FROZEN_DATETIME = datetime.strptime(dt, "%Y-%m-%d")
    elif isinstance(dt, datetime):
        _FROZEN_DATETIME = dt
    elif isinstance(dt, date):
        _FROZEN_DATETIME = datetime(dt.year, dt.month, dt.day)


def get_current_datetime(tz_name: Optional[str] = None) -> datetime:
    """Returns the current datetime in the specified timezone (or frozen clock if active)."""
    if _FROZEN_DATETIME is not None:
        return _FROZEN_DATETIME
    try:
        tz = zoneinfo.ZoneInfo(tz_name or DEFAULT_TIMEZONE)
        return datetime.now(tz)
    except Exception:
        return datetime.now()


def get_current_date(tz_name: Optional[str] = None) -> date:
    """Returns the current date (YYYY-MM-DD) in the specified timezone."""
    return get_current_datetime(tz_name).date()


def get_current_date_str(tz_name: Optional[str] = None) -> str:
    """Returns the current date as ISO string 'YYYY-MM-DD'."""
    return get_current_date(tz_name).strftime("%Y-%m-%d")


def format_relative_slot_datetime(
    slot_date_str: str,
    time_str: str,
    reference_dt: Optional[datetime] = None
) -> str:
    """
    Dynamically computes the human-readable datetime string for an appointment slot.
    Compares slot_date with the reference date (or current clock):
    - Same day: 'Today, Oct 19 at 10:00 AM'
    - Next day: 'Tomorrow, Oct 20 at 10:00 AM'
    - Otherwise: 'Oct 23 at 10:00 AM'
    """
    now = reference_dt if reference_dt is not None else get_current_datetime()
    ref_date = now.date()

    try:
        parts = slot_date_str.split("-")
        slot_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return f"{slot_date_str} at {time_str}"

    diff_days = (slot_date - ref_date).days
    month_name = slot_date.strftime("%b")
    day_num = slot_date.day

    if diff_days == 0:
        return f"Today, {month_name} {day_num} at {time_str}"
    elif diff_days == 1:
        return f"Tomorrow, {month_name} {day_num} at {time_str}"
    elif diff_days == -1:
        return f"Yesterday, {month_name} {day_num} at {time_str}"
    else:
        return f"{month_name} {day_num} at {time_str}"
