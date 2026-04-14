import pandas as pd

MARKET = "Netherlands"
CSV_PATH = "data/gsc_data.csv"


def load_gsc_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load Google Search Console query data.

    Expected columns:
    - Date
    - Query
    - Impressions
    """
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Rename to consistent internal names
    df = df.rename(columns={
        "date": "date",
        "query": "query",
        "impressions": "impressions",
    })

    # Parse date and numeric fields
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0)

    # Drop invalid rows
    df = df.dropna(subset=["date"]).copy()

    return df


def aggregate_daily_impressions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all query rows into one row per day.
    """
    daily = (
        df.groupby("date", as_index=False)["impressions"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def calculate_trend(daily_df: pd.DataFrame) -> dict:
    """
    Compare last 7 days vs previous 7 days.
    Requires at least 14 unique daily rows.
    """
    if len(daily_df) < 14:
        raise ValueError("GSC data must contain at least 14 unique daily rows.")

    last_7 = daily_df.tail(7)
    prev_7 = daily_df.iloc[-14:-7]

    current_impressions = last_7["impressions"].sum()
    previous_impressions = prev_7["impressions"].sum()

    if previous_impressions == 0:
        impression_change_pct = 0.0
    else:
        impression_change_pct = (
            (current_impressions - previous_impressions) / previous_impressions
        ) * 100

    return {
        "current_impressions": int(current_impressions),
        "previous_impressions": int(previous_impressions),
        "impression_change_pct": round(impression_change_pct, 1),
    }


def score_trend(impression_change_pct: float) -> tuple[int, str]:
    """
    Convert generic search trend into a score (0-30).
    """
    if impression_change_pct > 15:
        return 30, "Rising"
    elif impression_change_pct > 5:
        return 20, "Slightly Rising"
    elif impression_change_pct > -5:
        return 10, "Stable"
    else:
        return 5, "Declining"


def get_top_queries(df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """
    Return the top queries by total impressions.
    """
    query_summary = (
        df.groupby("query", as_index=False)["impressions"]
        .sum()
        .sort_values("impressions", ascending=False)
    )

    return query_summary.head(top_n)["query"].tolist()


def get_top_query_details(df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Return the top GSC queries in the last 7 days with period-over-period change.
    """
    unique_dates = df["date"].dropna().sort_values().unique()
    if len(unique_dates) < 14:
        return []

    last_7_dates = unique_dates[-7:]
    prev_7_dates = unique_dates[-14:-7]

    current_period = df[df["date"].isin(last_7_dates)]
    previous_period = df[df["date"].isin(prev_7_dates)]

    current_summary = (
        current_period.groupby("query", as_index=False)["impressions"]
        .sum()
        .rename(columns={"impressions": "current_impressions"})
    )
    previous_summary = (
        previous_period.groupby("query", as_index=False)["impressions"]
        .sum()
        .rename(columns={"impressions": "previous_impressions"})
    )
    total_summary = (
        df.groupby("query", as_index=False)["impressions"]
        .sum()
        .rename(columns={"impressions": "total_impressions"})
    )

    summary = (
        total_summary.merge(current_summary, on="query", how="left")
        .merge(previous_summary, on="query", how="left")
        .fillna(0)
    )

    summary["current_impressions"] = summary["current_impressions"].astype(int)
    summary["previous_impressions"] = summary["previous_impressions"].astype(int)
    summary["total_impressions"] = summary["total_impressions"].astype(int)

    def compute_change(row: pd.Series):
        previous = row["previous_impressions"]
        current = row["current_impressions"]
        if previous == 0:
            return None if current == 0 else "new"
        return round(((current - previous) / previous) * 100, 1)

    summary["change_pct"] = summary.apply(compute_change, axis=1)
    summary = summary.sort_values(
        ["current_impressions", "total_impressions"],
        ascending=[False, False],
    )

    top_queries = summary.head(top_n)
    return [
        {
            "query": row["query"],
            "current_impressions": int(row["current_impressions"]),
            "previous_impressions": int(row["previous_impressions"]),
            "change_pct": row["change_pct"],
        }
        for _, row in top_queries.iterrows()
    ]


def get_gsc_signal(csv_path: str = CSV_PATH) -> dict:
    """
    Main function used by the app.
    """
    df = load_gsc_data(csv_path)
    daily = aggregate_daily_impressions(df)
    trend = calculate_trend(daily)
    score, label = score_trend(trend["impression_change_pct"])
    top_queries = get_top_queries(df)
    top_query_details = get_top_query_details(df)

    return {
        "market": MARKET,
        "current_impressions": trend["current_impressions"],
        "previous_impressions": trend["previous_impressions"],
        "impression_change_pct": trend["impression_change_pct"],
        "trend_score": score,
        "trend_label": label,
        "top_queries": top_queries,
        "top_query_details": top_query_details,
        "daily_data": daily.to_dict(orient="records"),
    }


if __name__ == "__main__":
    result = get_gsc_signal()
    print(result)
