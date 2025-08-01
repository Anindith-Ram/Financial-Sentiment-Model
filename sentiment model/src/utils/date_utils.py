"""Date/time helper utilities shared across ingestion scripts."""

from datetime import date, timedelta, datetime
import re
from typing import Tuple

_WINDOW_RE = re.compile(r"^(\d+)([dmy])$")


def resolve_date_range(start_date: str = None,
                       end_date: str = None,
                       window: str = None,
                       default_window: str = "2y") -> Tuple[str, str, str]:
    """Resolve start/end dates based on explicit params or rolling window.

    Returns (start, end, window_used)
    where dates are ISO YYYY-MM-DD, window_used is the effective window string.
    """
    today = date.today()

    if window and (start_date or end_date):
        raise ValueError("Specify either --window or --start_date/--end_date, not both")

    if not window and not start_date:
        window = default_window

    if window:
        m = _WINDOW_RE.match(window)
        if not m:
            raise ValueError("Window must be like 90d, 18m, 2y")
        amount, unit = int(m.group(1)), m.group(2)
        if unit == "d":
            delta = timedelta(days=amount)
        elif unit == "m":
            delta = timedelta(days=amount * 30)
        else:  # years
            delta = timedelta(days=amount * 365)
        start = today - delta
        end = today
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), window

    # explicit dates
    if not end_date:
        end_date = today.strftime("%Y-%m-%d")
    return start_date, end_date, "explicit"


def add_window_cli(parser):
    """Inject --window, --start_date, --end_date options into argparse parser."""
    parser.add_argument('--window', type=str, help='Rolling window like 2y, 180d, 18m')
    parser.add_argument('--start_date', type=str, help='Explicit start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='Explicit end date YYYY-MM-DD')

# ---------------- After-hours helper -----------------

import pandas as _pd
import pytz as _pytz
from datetime import time as _time

_ET = _pytz.timezone("America/New_York")
_AFTER_CLOSE = _time(16, 0)   # 4:00 PM local
_BEFORE_OPEN = _time(9, 30)   # 9:30 AM local (next day)


def is_after_hours(ts_utc: _pd.Timestamp) -> bool:
    """Return True if timestamp (UTC) falls in US after-hours window.

    Window: 16:00-24:00 local OR 00:00-09:30 local (next session).
    Accepts naive or tz-aware UTC timestamps.
    """
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=_pytz.UTC)
    ts_et = ts_utc.astimezone(_ET)
    t = ts_et.time()
    return t >= _AFTER_CLOSE or t < _BEFORE_OPEN 