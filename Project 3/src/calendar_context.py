# ============================================================
# CALENDAR CONTEXT — 2024 holidays and Delhi NCR events
# ============================================================
# User-supplied reference data for Delhi NCR, 2024. Used to give
# the dashboard's date picker holiday/event context alongside the
# rainfall proxy — not verified against an external source by this
# pipeline; treat as given.

HOLIDAYS_2024 = {
    "2024-01-26": "Republic Day",
    "2024-03-25": "Holi",
    "2024-04-11": "Eid al-Fitr",
    "2024-08-15": "Independence Day",
    "2024-10-02": "Gandhi Jayanti",
    "2024-10-12": "Dussehra",
    "2024-10-31": "Diwali",
    "2024-11-01": "Diwali holiday",
}

EVENTS_2024 = {
    "2024-01-26": "Republic Day Parade",
    "2024-11-14": "India International Trade Fair",
    "2024-11-15": "India International Trade Fair",
    "2024-11-16": "India International Trade Fair",
    "2024-11-17": "India International Trade Fair",
    "2024-11-18": "India International Trade Fair",
    "2024-11-19": "India International Trade Fair",
    "2024-11-20": "India International Trade Fair",
    "2024-11-21": "India International Trade Fair",
    "2024-11-22": "India International Trade Fair",
    "2024-11-23": "India International Trade Fair",
    "2024-11-24": "India International Trade Fair",
    "2024-11-25": "India International Trade Fair",
    "2024-11-26": "India International Trade Fair",
    "2024-11-27": "India International Trade Fair",
}


def add_calendar_context(df):
    """Adds 'holiday' and 'event' columns to df based on its 'date' column."""
    import pandas as pd

    dates = pd.to_datetime(df["date"])
    date_keys = dates.dt.strftime("%Y-%m-%d")
    df = df.copy()
    df["holiday"] = date_keys.map(HOLIDAYS_2024).fillna("No holiday")
    df["event"] = date_keys.map(EVENTS_2024).fillna("No scheduled event")
    return df
