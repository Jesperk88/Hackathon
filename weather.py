import requests

LAT = 52.0907
LON = 5.1214
MARKET = "Netherlands"


def fetch_weather(lat: float = LAT, lon: float = LON):
    """
    Fetch 3-day weather forecast for a given location.
    Returns raw JSON.
    """
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum",
        "forecast_days": 3,
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


def extract_weather_metrics(weather_json: dict):
    """
    Extract average temperature and total rain for the next few days.
    """
    daily = weather_json.get("daily", {})

    temps = daily.get("temperature_2m_max", [])
    rain = daily.get("precipitation_sum", [])

    if not temps or not rain:
        return {
            "avg_temp": None,
            "total_rain": None
        }

    avg_temp = round(sum(temps) / len(temps), 1)
    total_rain = round(sum(rain), 1)

    return {
        "avg_temp": avg_temp,
        "total_rain": total_rain
    }


def calculate_weather_score(avg_temp: float, total_rain: float):
    """
    Score weather between 0–30 based on conditions favorable for bookings.
    """
    score = 0

    if avg_temp is not None:
        if 18 <= avg_temp <= 26:
            score += 15
        elif 12 <= avg_temp < 18:
            score += 10
        elif avg_temp > 26:
            score += 8
        else:
            score += 4

    if total_rain is not None:
        if total_rain < 1:
            score += 15
        elif total_rain < 5:
            score += 8
        else:
            score += 2

    return min(score, 30)


def get_weather_label(score: int):
    if score >= 24:
        return "Favourable"
    elif score >= 16:
        return "Mixed"
    return "Unfavourable"


def get_weather_summary(avg_temp: float, total_rain: float, label: str):
    if avg_temp is None or total_rain is None:
        return "Weather data unavailable."

    if label == "Favourable":
        return "Mild temperatures and low rainfall support booking demand."
    elif label == "Mixed":
        return "Weather conditions are mixed and may have limited impact on demand."
    else:
        return "Cool or wet weather may reduce short-term booking demand."


def get_weather_signal(lat: float = LAT, lon: float = LON):
    """
    End-to-end function:
    fetch → extract → score → label → summary
    """
    data = fetch_weather(lat, lon)
    metrics = extract_weather_metrics(data)

    score = calculate_weather_score(
        metrics["avg_temp"],
        metrics["total_rain"]
    )

    label = get_weather_label(score)
    summary = get_weather_summary(
        metrics["avg_temp"],
        metrics["total_rain"],
        label
    )

    return {
        "market": MARKET,
        "avg_temp": metrics["avg_temp"],
        "total_rain": metrics["total_rain"],
        "weather_score": score,
        "weather_label": label,
        "weather_summary": summary
    }


if __name__ == "__main__":
    result = get_weather_signal()
    print("Weather Signal:")
    print(result)