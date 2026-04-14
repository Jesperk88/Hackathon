import streamlit as st
import pandas as pd
from scoring import evaluate_market

st.set_page_config(page_title="Predictive Signal Framework", layout="wide")


def safe_get(data: dict, key: str, default="N/A"):
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def format_change(change):
    if change == "new":
        return "new"
    if change is None:
        return "n/a"
    return f"{change:+.1f}%"


result = evaluate_market()

weather = safe_get(result, "weather", {})
ads = safe_get(result, "ads", {})
gsc = safe_get(result, "gsc", {})

st.title("Predictive Signal Framework")
st.subheader("Netherlands Demand Steering Prototype")

top_col1, top_col2, top_col3 = st.columns(3)

with top_col1:
    st.metric("Total Score", safe_get(result, "total_score", "N/A"), delta="High demand")

with top_col2:
    st.metric("Market State", safe_get(result, "state", "N/A"))

with top_col3:
    st.metric("Confidence", safe_get(result, "confidence", "N/A"))

st.markdown("## 🚀 Recommendation")
st.success(safe_get(result, "recommendation", "No recommendation available."))

# PRIORITY 1 — Why this recommendation?
st.markdown("### Why?")
st.info(
    f"""
Demand is **{safe_get(gsc, 'trend_label')} (+{safe_get(gsc, 'impression_change_pct')}%)**,
weather conditions are **{safe_get(weather, 'weather_label')}**,
and paid demand is **{safe_get(ads, 'paid_label')} (+{safe_get(ads, 'impression_change_pct')}%)**.

→ This indicates a strong increase in demand for the coming days.
"""
)

# PRIORITY 8 — AI-like insight
st.markdown("### Insight")
st.write(
    "Demand is expected to increase due to strong search growth and favorable weather conditions."
)


st.markdown("## Signal Breakdown")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌦 Weather")
    st.write(f"Score: **{safe_get(weather, 'weather_score')} / 30**")
    st.write(f"Label: **{safe_get(weather, 'weather_label')}**")
    st.write(f"Average Temp: **{safe_get(weather, 'avg_temp')}°C**")
    st.write(f"Total Rain: **{safe_get(weather, 'total_rain')} mm**")
    st.write(safe_get(weather, "weather_summary", ""))
    st.caption("Good weather → higher likelihood of bookings")

with col2:
    st.markdown("### 📊 Google Ads")
    st.write(f"Score: **{safe_get(ads, 'paid_score')} / 30**")
    st.write(f"Label: **{safe_get(ads, 'paid_label')}**")
    st.write(f"Current Impressions: **{safe_get(ads, 'current_impressions')}**")
    st.write(f"Previous Impressions: **{safe_get(ads, 'previous_impressions')}**")
    st.write(f"Change: **{safe_get(ads, 'impression_change_pct')}%**")
    st.write(f"Current Clicks: **{safe_get(ads, 'current_clicks')}**")
    st.write(f"Current Cost: **€{safe_get(ads, 'current_cost')}**")
    st.write(f"Current Conversions: **{safe_get(ads, 'current_conversions')}**")

    top_search_term_details = safe_get(ads, "top_search_term_details", [])
    if isinstance(top_search_term_details, list) and top_search_term_details:
        st.write("Top search terms:")
        for item in top_search_term_details:
            term = item.get("term", "Unknown term")
            change = format_change(item.get("change_pct"))
            st.write(f"- {term} ({change})")

    st.caption("Rising impressions → increasing market demand")

with col3:
    st.markdown("### 🔍 Search Console")
    st.write(f"Score: **{safe_get(gsc, 'trend_score')} / 30**")
    st.write(f"Label: **{safe_get(gsc, 'trend_label')}**")
    st.write(f"Current Impressions: **{safe_get(gsc, 'current_impressions')}**")
    st.write(f"Previous Impressions: **{safe_get(gsc, 'previous_impressions')}**")
    st.write(f"Change: **{safe_get(gsc, 'impression_change_pct')}%**")

    top_query_details = safe_get(gsc, "top_query_details", [])
    if isinstance(top_query_details, list) and top_query_details:
        st.write("Top queries:")
        for item in top_query_details:
            query = item.get("query", "Unknown query")
            change = format_change(item.get("change_pct"))
            st.write(f"- {query} ({change})")

    st.caption("Search growth → early demand indicator")

# PRIORITY 2 — Trend visualization
if "daily_data" in gsc:
    df = pd.DataFrame(gsc["daily_data"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        st.markdown("## Demand Trend (Search)")
        st.line_chart(df.set_index("date")["impressions"])

st.markdown("## Interpretation")
st.info(
    "This prototype combines external weather conditions with paid and organic "
    "search demand signals to support more proactive marketing budget decisions."
)
