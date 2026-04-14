from pathlib import Path
import pandas as pd
import requests
import matplotlib.pyplot as plt


# -------------------------------
# Configuration
# -------------------------------
BOOKING_START_DATE = "2023-01-01"
LAT = 52.0907   # Utrecht / Netherlands proxy
LON = 5.1214

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BOOKING_FILE = DATA_DIR / "booking.csv"   # change if needed
OUTPUT_FILE = DATA_DIR / "bookings_weather_comparison.csv"

LAGS_TO_TEST = [0, 1, 2, 3, 7]


# -------------------------------
# Load bookings data
# -------------------------------
def load_bookings_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find booking file: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "booking_date" not in df.columns:
        raise ValueError("Expected a 'booking_date' column in the booking CSV.")

    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df = df.dropna(subset=["booking_date"]).copy()

    df = df[df["booking_date"] >= pd.Timestamp(BOOKING_START_DATE)]
    df = df.sort_values("booking_date").reset_index(drop=True)

    numeric_cols = ["bookings", "rental_revenue", "guests", "booked_nights", "avg_adr"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -------------------------------
# Fetch historical weather data
# -------------------------------
def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    daily = data.get("daily", {})
    weather = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", []), errors="coerce"),
        "temp_max": daily.get("temperature_2m_max", []),
        "temp_min": daily.get("temperature_2m_min", []),
        "temp_mean": daily.get("temperature_2m_mean", []),
        "rain_mm": daily.get("precipitation_sum", []),
    })

    return weather


# -------------------------------
# Merge bookings + weather
# -------------------------------
def merge_bookings_weather(bookings: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = bookings.merge(
        weather,
        left_on="booking_date",
        right_on="date",
        how="inner"
    )

    merged = merged.drop(columns=["date"]).copy()
    merged = merged.sort_values("booking_date").reset_index(drop=True)

    return merged


# -------------------------------
# Feature engineering
# -------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["year"] = enriched["booking_date"].dt.year
    enriched["month"] = enriched["booking_date"].dt.month
    enriched["day_of_week"] = enriched["booking_date"].dt.dayofweek
    enriched["week_of_year"] = enriched["booking_date"].dt.isocalendar().week.astype(int)

    return enriched


def add_normalized_bookings(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    if "bookings" not in enriched.columns:
        raise ValueError("Expected a 'bookings' column for normalization.")

    month_avg = enriched.groupby("month")["bookings"].transform("mean")
    dow_avg = enriched.groupby("day_of_week")["bookings"].transform("mean")

    enriched["bookings_month_avg"] = month_avg
    enriched["bookings_dow_avg"] = dow_avg

    enriched["bookings_vs_month_avg"] = enriched["bookings"] / month_avg
    enriched["bookings_vs_dow_avg"] = enriched["bookings"] / dow_avg

    return enriched


# -------------------------------
# Correlation analysis
# -------------------------------
def print_raw_correlations(df: pd.DataFrame) -> None:
    print("\n--- Raw correlation matrix ---")
    cols_to_check = ["bookings", "temp_mean", "temp_max", "rain_mm"]
    available = [c for c in cols_to_check if c in df.columns]

    corr = df[available].corr(numeric_only=True)
    print(corr)

    if "bookings" in df.columns and "temp_mean" in df.columns:
        print(
            "\nRaw correlation between bookings and average temperature:",
            round(df["bookings"].corr(df["temp_mean"]), 3)
        )

    if "bookings" in df.columns and "rain_mm" in df.columns:
        print(
            "Raw correlation between bookings and rainfall:",
            round(df["bookings"].corr(df["rain_mm"]), 3)
        )


def print_normalized_correlations(df: pd.DataFrame) -> None:
    print("\n--- Normalized / seasonality-aware correlations ---")

    if "bookings_vs_month_avg" in df.columns and "temp_mean" in df.columns:
        print(
            "Correlation between normalized bookings (vs monthly avg) and average temperature:",
            round(df["bookings_vs_month_avg"].corr(df["temp_mean"]), 3)
        )

    if "bookings_vs_month_avg" in df.columns and "rain_mm" in df.columns:
        print(
            "Correlation between normalized bookings (vs monthly avg) and rainfall:",
            round(df["bookings_vs_month_avg"].corr(df["rain_mm"]), 3)
        )

    if "bookings_vs_dow_avg" in df.columns and "temp_mean" in df.columns:
        print(
            "Correlation between normalized bookings (vs weekday avg) and average temperature:",
            round(df["bookings_vs_dow_avg"].corr(df["temp_mean"]), 3)
        )

    if "bookings_vs_dow_avg" in df.columns and "rain_mm" in df.columns:
        print(
            "Correlation between normalized bookings (vs weekday avg) and rainfall:",
            round(df["bookings_vs_dow_avg"].corr(df["rain_mm"]), 3)
        )


def print_monthly_weather_booking_summary(df: pd.DataFrame) -> None:
    print("\n--- Monthly summary (seasonality check) ---")
    monthly = (
        df.groupby("month", as_index=False)
        .agg({
            "bookings": "mean",
            "temp_mean": "mean",
            "rain_mm": "mean"
        })
        .sort_values("month")
    )
    print(monthly)


# -------------------------------
# Lag analysis
# -------------------------------
def run_lag_analysis(df: pd.DataFrame, target_col: str, label: str) -> pd.DataFrame:
    lag_results = []

    for lag in LAGS_TO_TEST:
        temp_lag_col = f"temp_mean_lag_{lag}"
        rain_lag_col = f"rain_mm_lag_{lag}"

        df[temp_lag_col] = df["temp_mean"].shift(lag)
        df[rain_lag_col] = df["rain_mm"].shift(lag)

        temp_corr = df[target_col].corr(df[temp_lag_col])
        rain_corr = df[target_col].corr(df[rain_lag_col])

        lag_results.append({
            "lag_days": lag,
            "temp_corr": temp_corr,
            "rain_corr": rain_corr,
        })

    lag_df = pd.DataFrame(lag_results)

    print(f"\n--- Lag analysis ({label}) ---")
    print(lag_df)

    return lag_df


# -------------------------------
# Plotting
# -------------------------------
def create_plots(df: pd.DataFrame) -> None:
    # Bookings over time
    plt.figure(figsize=(12, 5))
    plt.plot(df["booking_date"], df["bookings"])
    plt.title("Bookings over time")
    plt.xlabel("Date")
    plt.ylabel("Bookings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Temperature over time
    plt.figure(figsize=(12, 5))
    plt.plot(df["booking_date"], df["temp_mean"])
    plt.title("Average temperature over time")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Scatter: temp vs bookings
    plt.figure(figsize=(8, 5))
    plt.scatter(df["temp_mean"], df["bookings"], alpha=0.5)
    plt.title("Bookings vs Average Temperature")
    plt.xlabel("Average Temperature (°C)")
    plt.ylabel("Bookings")
    plt.tight_layout()
    plt.show()

    # Scatter: rain vs bookings
    plt.figure(figsize=(8, 5))
    plt.scatter(df["rain_mm"], df["bookings"], alpha=0.5)
    plt.title("Bookings vs Rainfall")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Bookings")
    plt.tight_layout()
    plt.show()

    # Scatter: temp vs normalized bookings
    plt.figure(figsize=(8, 5))
    plt.scatter(df["temp_mean"], df["bookings_vs_month_avg"], alpha=0.5)
    plt.title("Normalized Bookings (vs Monthly Avg) vs Average Temperature")
    plt.xlabel("Average Temperature (°C)")
    plt.ylabel("Normalized bookings")
    plt.tight_layout()
    plt.show()

    # Scatter: rain vs normalized bookings
    plt.figure(figsize=(8, 5))
    plt.scatter(df["rain_mm"], df["bookings_vs_month_avg"], alpha=0.5)
    plt.title("Normalized Bookings (vs Monthly Avg) vs Rainfall")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Normalized bookings")
    plt.tight_layout()
    plt.show()

    # Monthly averages
    monthly = df.copy()
    monthly["month_date"] = monthly["booking_date"].dt.to_period("M").dt.to_timestamp()
    monthly = monthly.groupby("month_date", as_index=False).agg({
        "bookings": "mean",
        "temp_mean": "mean",
        "rain_mm": "mean"
    })

    plt.figure(figsize=(12, 5))
    plt.plot(monthly["month_date"], monthly["bookings"], label="Avg bookings")
    plt.plot(monthly["month_date"], monthly["temp_mean"], label="Avg temperature")
    plt.title("Monthly average bookings and temperature")
    plt.xlabel("Month")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -------------------------------
# Holiday payout check
# -------------------------------
def print_holiday_payout_window_effect(df: pd.DataFrame) -> None:
    if "bookings" not in df.columns or "booking_date" not in df.columns:
        print("\n--- Holiday payout window check skipped: missing required columns ---")
        return

    analysis = df.copy()
    analysis["day"] = analysis["booking_date"].dt.day
    analysis["is_payout_window"] = (
        ((analysis["month"] == 5) & (analysis["day"] >= 25))
        | ((analysis["month"] == 6) & (analysis["day"] <= 7))
    )

    payout_data = analysis[analysis["is_payout_window"]]["bookings"].dropna()
    baseline_data = analysis[~analysis["is_payout_window"]]["bookings"].dropna()

    if payout_data.empty or baseline_data.empty:
        print("\n--- Holiday payout window check skipped: insufficient data ---")
        return

    payout_avg = payout_data.mean()
    baseline_avg = baseline_data.mean()
    pct_change = ((payout_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg else float("nan")
    baseline_std = baseline_data.std()
    effect_size = ((payout_avg - baseline_avg) / baseline_std) if baseline_std else float("nan")

    weekly = (
        analysis.assign(
            week_start=analysis["booking_date"].dt.to_period("W-MON").dt.start_time
        )
        .groupby(["week_start", "is_payout_window"], as_index=False)["bookings"]
        .mean()
    )
    payout_weekly = weekly[weekly["is_payout_window"]]["bookings"]
    baseline_weekly = weekly[~weekly["is_payout_window"]]["bookings"]

    print("\n--- Holiday payout window check (last week of May + first week of June) ---")
    print(f"Average bookings in payout window: {payout_avg:.2f}")
    print(f"Average bookings in other periods: {baseline_avg:.2f}")
    print(f"Difference: {pct_change:.1f}%")

    if not payout_weekly.empty and not baseline_weekly.empty:
        print(f"Average weekly bookings in payout window: {payout_weekly.mean():.2f}")
        print(f"Average weekly bookings in other weeks: {baseline_weekly.mean():.2f}")

    if pd.notna(effect_size) and effect_size >= 0.5:
        print(
            "Interpretation: The payout window shows a potentially significant uplift in "
            "bookings compared with other weeks."
        )
    else:
        print(
            "Interpretation: The payout window does not show a clearly significant uplift "
            "in bookings compared with other weeks."
        )


# -------------------------------
# Main
# -------------------------------
def main():
    print("Loading bookings data...")
    bookings = load_bookings_data(BOOKING_FILE)

    print("Bookings loaded:")
    print(bookings.head())
    print(f"Rows: {len(bookings)}")

    end_date = bookings["booking_date"].max().strftime("%Y-%m-%d")

    print(f"\nFetching weather data from {BOOKING_START_DATE} to {end_date}...")
    weather = fetch_historical_weather(
        lat=LAT,
        lon=LON,
        start_date=BOOKING_START_DATE,
        end_date=end_date,
    )

    print("Weather loaded:")
    print(weather.head())
    print(f"Rows: {len(weather)}")

    print("\nMerging bookings and weather...")
    comparison = merge_bookings_weather(bookings, weather)
    comparison = add_time_features(comparison)
    comparison = add_normalized_bookings(comparison)

    print("Merged data:")
    print(comparison.head())
    print(f"Rows: {len(comparison)}")

    print_raw_correlations(comparison)
    print_normalized_correlations(comparison)
    print_monthly_weather_booking_summary(comparison)
    print_holiday_payout_window_effect(comparison)

    run_lag_analysis(comparison.copy(), target_col="bookings", label="raw bookings")
    run_lag_analysis(
        comparison.copy(),
        target_col="bookings_vs_month_avg",
        label="normalized bookings vs monthly average"
    )

    comparison.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMerged dataset saved to: {OUTPUT_FILE}")

    create_plots(comparison)


if __name__ == "__main__":
    main()
