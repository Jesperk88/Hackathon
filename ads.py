import pandas as pd

MARKET = "Netherlands"
CSV_PATH = "data/ads_data.csv"


def load_ads_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load Google Ads search term data.
    Expected columns:
    - Day
    - Clicks
    - Impr.
    - Cost
    - Conversions
    """

    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Rename key columns
    df = df.rename(columns={
        "day": "date",
        "impr.": "impressions",
        "cost": "cost",
        "clicks": "clicks",
        "conversions": "conversions",
    })

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Clean numeric fields
    numeric_cols = ["clicks", "impressions", "cost", "conversions"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Drop bad dates if any
    df = df.dropna(subset=["date"]).copy()

    return df


def aggregate_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all search-term rows into one row per day.
    """
    daily = (
        df.groupby("date", as_index=False)[["clicks", "impressions", "cost", "conversions"]]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def calculate_period_comparison(daily_df: pd.DataFrame) -> dict:
    """
    Compare last 7 days vs previous 7 days.
    Requires at least 14 unique days in the CSV.
    """
    if len(daily_df) < 14:
        raise ValueError("Ads data must contain at least 14 unique daily rows.")

    last_7 = daily_df.tail(7)
    prev_7 = daily_df.iloc[-14:-7]

    current_impressions = last_7["impressions"].sum()
    previous_impressions = prev_7["impressions"].sum()

    current_clicks = last_7["clicks"].sum()
    previous_clicks = prev_7["clicks"].sum()

    current_cost = last_7["cost"].sum()
    previous_cost = prev_7["cost"].sum()

    current_conversions = last_7["conversions"].sum()
    previous_conversions = prev_7["conversions"].sum()

    if previous_impressions == 0:
        impression_change_pct = 0.0
    else:
        impression_change_pct = (
            (current_impressions - previous_impressions) / previous_impressions
        ) * 100

    return {
        "current_impressions": int(current_impressions),
        "previous_impressions": int(previous_impressions),
        "current_clicks": int(current_clicks),
        "previous_clicks": int(previous_clicks),
        "current_cost": round(current_cost, 2),
        "previous_cost": round(previous_cost, 2),
        "current_conversions": round(current_conversions, 2),
        "previous_conversions": round(previous_conversions, 2),
        "impression_change_pct": round(impression_change_pct, 1),
    }


def get_top_search_terms(df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """
    Return the top Google Ads search terms by total impressions.
    """
    if "search term" not in df.columns:
        return []

    filtered = df.copy()
    filtered["search term"] = filtered["search term"].astype(str).str.strip()
    filtered = filtered[
        filtered["search term"].ne("")
        & ~filtered["search term"].str.startswith("Total:", na=False)
    ]

    term_summary = (
        filtered.groupby("search term", as_index=False)["impressions"]
        .sum()
        .sort_values("impressions", ascending=False)
    )

    return term_summary.head(top_n)["search term"].tolist()


def score_paid_demand(impression_change_pct: float) -> tuple[int, str]:
    """
    Convert paid search demand trend into a score (0-30).
    """
    if impression_change_pct > 15:
        return 30, "Rising"
    elif impression_change_pct > 5:
        return 20, "Slightly Rising"
    elif impression_change_pct > -5:
        return 10, "Stable"
    else:
        return 5, "Declining"


def get_ads_signal(csv_path: str = CSV_PATH) -> dict:
    """
    Main function used by the app.
    """
    df = load_ads_data(csv_path)
    daily = aggregate_daily_metrics(df)
    comparison = calculate_period_comparison(daily)
    score, label = score_paid_demand(comparison["impression_change_pct"])
    top_search_terms = get_top_search_terms(df)

    return {
        "market": MARKET,
        "current_impressions": comparison["current_impressions"],
        "previous_impressions": comparison["previous_impressions"],
        "current_clicks": comparison["current_clicks"],
        "previous_clicks": comparison["previous_clicks"],
        "current_cost": comparison["current_cost"],
        "previous_cost": comparison["previous_cost"],
        "current_conversions": comparison["current_conversions"],
        "previous_conversions": comparison["previous_conversions"],
        "impression_change_pct": comparison["impression_change_pct"],
        "top_search_terms": top_search_terms,
        "paid_score": score,
        "paid_label": label,
    }


if __name__ == "__main__":
    result = get_ads_signal()
    print(result)
