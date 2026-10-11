"""
Stage 4: Live Meteorological & Topographical Gradient API Fusion.
Retrieves hourly Wind Speed, Wind Direction, Relative Humidity, Temperature,
and 5-point Digital Elevation Model (DEM) slope gradients from Open-Meteo APIs.
"""

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Dict, List
import httpx


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"


def compass_direction_label(deg: float) -> str:
    """Convert meteorological wind direction (degrees where wind blows FROM) to compass cardinal."""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int(round(deg / 22.5)) % 16
    return dirs[ix]


async def fetch_weather_forecast(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch current + next 12 hours of wind speed, wind direction, relative humidity,
    and temperature from the official Open-Meteo Forecast API.
    """
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "kmh",
        "forecast_days": 2,
        "timezone": "UTC",
    }

    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.get(OPEN_METEO_FORECAST_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    current = data.get("current", {})
    hourly = data.get("hourly", {})
    times: List[str] = hourly.get("time", [])
    temps: List[float] = hourly.get("temperature_2m", [])
    rh_list: List[float] = hourly.get("relative_humidity_2m", [])
    ws_list: List[float] = hourly.get("wind_speed_10m", [])
    wd_list: List[float] = hourly.get("wind_direction_10m", [])

    now_hour_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:")
    start_idx = 0
    for idx, t_str in enumerate(times):
        if t_str.startswith(now_hour_prefix):
            start_idx = idx
            break

    hourly_12h: List[Dict[str, Any]] = []
    for step in range(12):
        i = min(start_idx + step, max(0, len(times) - 1))
        if i < len(times):
            w_from = float(wd_list[i]) if i < len(wd_list) and wd_list[i] is not None else 240.0
            # Wind blows FROM w_from degrees, so fire is pushed TOWARD (w_from + 180) % 360
            w_toward = (w_from + 180.0) % 360.0
            hourly_12h.append(
                {
                    "hour_offset": step + 1,
                    "time_utc": times[i] + "Z",
                    "temperature_c": round(float(temps[i]), 1) if i < len(temps) and temps[i] is not None else 24.0,
                    "relative_humidity_pct": round(float(rh_list[i]), 1) if i < len(rh_list) and rh_list[i] is not None else 35.0,
                    "wind_speed_kmh": round(float(ws_list[i]), 1) if i < len(ws_list) and ws_list[i] is not None else 18.0,
                    "wind_from_deg": round(w_from, 0),
                    "wind_toward_deg": round(w_toward, 0),
                }
            )

    curr_w_from = float(current.get("wind_direction_10m") or (hourly_12h[0]["wind_from_deg"] if hourly_12h else 240.0))
    curr_w_toward = (curr_w_from + 180.0) % 360.0
    curr_ws = float(current.get("wind_speed_10m") or (hourly_12h[0]["wind_speed_kmh"] if hourly_12h else 18.0))
    curr_rh = float(current.get("relative_humidity_2m") or (hourly_12h[0]["relative_humidity_pct"] if hourly_12h else 35.0))
    curr_temp = float(current.get("temperature_2m") or (hourly_12h[0]["temperature_c"] if hourly_12h else 25.0))
    curr_gust = float(current.get("wind_gusts_10m") or curr_ws * 1.25)

    return {
        "source": "Open-Meteo Meteorological API",
        "current": {
            "temperature_c": round(curr_temp, 1),
            "relative_humidity_pct": round(curr_rh, 1),
            "wind_speed_kmh": round(curr_ws, 1),
            "wind_gusts_kmh": round(curr_gust, 1),
            "wind_from_deg": round(curr_w_from, 0),
            "wind_from_cardinal": compass_direction_label(curr_w_from),
            "wind_toward_deg": round(curr_w_toward, 0),
            "wind_toward_cardinal": compass_direction_label(curr_w_toward),
            "dryness_risk": (
                "CRITICAL (<25% RH)"
                if curr_rh < 25
                else ("HIGH (25-45% RH)" if curr_rh < 45 else "MODERATE (>45% RH)")
            ),
        },
        "hourly_12h": hourly_12h,
    }


async def fetch_topographical_gradient(lat: float, lon: float) -> Dict[str, Any]:
    """
    Query Open-Meteo Digital Elevation Model (DEM) API at 5 points (Center, North, South, East, West)
    spaced ~1.5 km apart to compute local terrain elevation, slope angle (degrees), and uphill direction!
    """
    delta_deg = 0.0135  # ~1.5 km spacing
    lats = [
        round(lat, 4),              # Center
        round(lat + delta_deg, 4),  # North (+1.5 km)
        round(lat - delta_deg, 4),  # South (-1.5 km)
        round(lat, 4),              # East
        round(lat, 4),              # West
    ]
    lons = [
        round(lon, 4),
        round(lon, 4),
        round(lon, 4),
        round(lon + delta_deg, 4),
        round(lon - delta_deg, 4),
    ]

    params = {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(OPEN_METEO_ELEVATION_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        elevs = data.get("elevation", [300.0] * 5)
        z_center, z_north, z_south, z_east, z_west = [float(z or 0.0) for z in elevs[:5]]

        # Compute gradient dz/dx (East-West) and dz/dy (North-South) over 3000 meters span
        span_ns_m = 3000.0
        span_ew_m = max(500.0, 3000.0 * math.cos(math.radians(lat)))

        dz_dy = (z_north - z_south) / span_ns_m  # + means uphill toward North
        dz_dx = (z_east - z_west) / span_ew_m    # + means uphill toward East

        slope_rad = math.atan(math.hypot(dz_dx, dz_dy))
        slope_deg = round(math.degrees(slope_rad), 2)

        # Uphill compass bearing (0 deg = North, 90 deg = East)
        uphill_bearing_deg = (math.degrees(math.atan2(dz_dx, dz_dy)) + 360.0) % 360.0

        return {
            "source": "Open-Meteo SRTM/Copernicus DEM Elevation API",
            "center_elevation_m": round(z_center, 1),
            "sample_elevations_m": {
                "center": round(z_center, 1),
                "north_1_5km": round(z_north, 1),
                "south_1_5km": round(z_south, 1),
                "east_1_5km": round(z_east, 1),
                "west_1_5km": round(z_west, 1),
            },
            "slope_degrees": slope_deg,
            "uphill_bearing_deg": round(uphill_bearing_deg, 0),
            "uphill_cardinal": compass_direction_label(uphill_bearing_deg),
        }
    except Exception:
        return {
            "source": "Open-Meteo DEM Elevation API (Fallback)",
            "center_elevation_m": 250.0,
            "slope_degrees": 3.5,
            "uphill_bearing_deg": 45.0,
            "uphill_cardinal": "NE",
        }


async def fetch_fused_environment(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch meteorological forecast and topographical slope gradient concurrently."""
    weather, terrain = await asyncio.gather(
        fetch_weather_forecast(lat, lon),
        fetch_topographical_gradient(lat, lon),
    )
    return {
        "weather": weather,
        "terrain": terrain,
    }
