import time
import pandas as pd
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder


def _get_timezone_for_location(
    state: str, country: str, geolocator: Nominatim, tf: TimezoneFinder
) -> str | None:
    """
    Geocode a state/country pair and return its IANA timezone string.

    :param state: State or region name (can be None).
    :param country: Country name.
    :param geolocator: Nominatim geocoder instance.
    :param tf: TimezoneFinder instance.
    :return: Timezone string (e.g. "Europe/Rome") or None if not found.
    """
    try:
        query = f"{state}, {country}" if pd.notnull(state) else country
        location = geolocator.geocode(query)
        print(f"Geocoding query: '{query}' -> Location: {location}", flush=True)
        if location:
            print(
                f"Found location for query '{query}': {location.address} "
                f"(lat: {location.latitude}, lng: {location.longitude})",
                flush=True,
            )
            return tf.timezone_at(lng=location.longitude, lat=location.latitude)
    except Exception as e:
        print(
            f"Error geocoding location for state '{state}' and country '{country}': {e}",
            flush=True,
        )
    return None


def find_timezones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a DataFrame with a 'timezone' column by geocoding unique (state, country) pairs.

    :param df: Input DataFrame containing 'state' and 'country' columns.
    :return: DataFrame with an additional 'timezone' column.
    """
    unique_locations = df[["state", "country"]].drop_duplicates()
    timezones_records = []

    tf = TimezoneFinder(in_memory=True)
    geolocator = Nominatim(user_agent="lt_project")

    for _, row in unique_locations.iterrows():
        state = row["state"]
        country = row["country"]
        timezone = _get_timezone_for_location(
            state=state, country=country, geolocator=geolocator, tf=tf
        )
        time.sleep(1)  # Respect geocoding service rate limits
        timezones_records.append(
            {"state": state, "country": country, "timezone": timezone}
        )

    timezones_df = pd.DataFrame(timezones_records)
    return df.merge(timezones_df, on=["state", "country"], how="left")


def normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'timestamp' column from UTC to each row's local timezone,
    adding a 'timestamp_local' column.

    :param df: DataFrame with 'timestamp' (UTC) and 'timezone' columns.
    :return: DataFrame with an additional 'timestamp_local' column.
    """
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df["timestamp"] = ts
    df["timestamp_local"] = df.apply(
        lambda row: row["timestamp"].tz_convert(row["timezone"]), axis=1
    )
    return df
