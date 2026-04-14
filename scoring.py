from weather import get_weather_signal
from ads import get_ads_signal
from gsc import get_gsc_signal


def calculate_total_score(weather_score: int, ads_score: int, gsc_score: int) -> int:
    return weather_score + ads_score + gsc_score


def classify_state(total_score: int) -> str:
    if total_score >= 70:
        return "Demand Surge"
    elif total_score >= 50:
        return "Opportunity"
    elif total_score >= 30:
        return "Stable"
    else:
        return "Weak Market"
    
def get_confidence(weather: dict, ads: dict, gsc: dict) -> str:
    strong_signals = 0

    if weather.get("weather_score", 0) > 20:
        strong_signals += 1
    if ads.get("paid_score", 0) > 20:
        strong_signals += 1
    if gsc.get("trend_score", 0) > 20:
        strong_signals += 1

    if strong_signals == 3:
        return "High"
    elif strong_signals == 2:
        return "Medium"
    else:
        return "Low"


def get_recommendation(state: str, ads: dict, gsc: dict) -> str:
    top_query = "generic high-intent keywords"
    if isinstance(gsc.get("top_queries"), list) and len(gsc["top_queries"]) > 0:
        top_query = gsc["top_queries"][0]

    gsc_change = gsc.get("impression_change_pct", 0)

    if state == "Demand Surge":
        return f"""Increase SEA budget by 15 to 20% in the Netherlands.

Prioritize high-intent keywords such as "{top_query}" and generic search campaigns.

Expected impact:
- Capture rising demand (+{gsc_change}% in generic search impressions)
"""
    elif state == "Opportunity":
        return f"""Maintain spend and shift budget toward high-intent campaigns.

Prioritize generic search themes with rising demand and keywords such as "{top_query}".

Expected impact:
- Capture growing demand more efficiently
"""
    elif state == "Stable":
        return """Keep budget stable and monitor changes closely.

Keep emphasis on core campaigns while monitoring search trends and weather shifts.

Expected impact:
- Preserve efficiency while monitoring for change
"""
    else:
        return """Reduce spend or reallocate budget to stronger themes or markets.

Protect efficiency and keep spend concentrated on the strongest-performing campaigns.

Expected impact:
- Lower wasted spend in weaker demand conditions
"""


def evaluate_market() -> dict:
    weather = get_weather_signal()
    ads = get_ads_signal()
    gsc = get_gsc_signal()

    total_score = calculate_total_score(
        weather_score=weather["weather_score"],
        ads_score=ads["paid_score"],
        gsc_score=gsc["trend_score"],
    )

    state = classify_state(total_score)
    recommendation = get_recommendation(state, ads, gsc)
    confidence = get_confidence(weather, ads, gsc)
    
    return {
        "market": "Netherlands",
        "weather": weather,
        "ads": ads,
        "gsc": gsc,
        "total_score": total_score,
        "state": state,
        "recommendation": recommendation,
        "confidence": confidence,
    }


if __name__ == "__main__":
    result = evaluate_market()
    print(result)
