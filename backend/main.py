"""
WildfireWatch Backend — Full Multi-API Disaster Perimeter & Spread Forecasting Engine.
Integrates:
1. NASA FIRMS Area REST API (Satellite thermal anomalies)
2. Spatial Clustering & Sensor-Footprint Alpha-Hull Boundary Extraction (Stage 2 & 3)
3. Element84 Earth Search STAC API (Sentinel-2 L2A Multispectral NIR/SWIR metadata)
4. Open-Meteo Meteorological Forecast API + DEM Elevation/Slope API (Stage 4)
5. Multi-Variable 6h & 12h Elliptical Wildfire Spread Forecasting Engine (Stage 5 & 6)
"""

import asyncio
import csv
import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from services.perimeter_engine import (
    cluster_hotspots_into_incidents,
    query_sentinel2_stac_metadata,
)
from services.spread_forecaster import simulate_spread_polygons
from services.unet_and_ml_engine import (
    RF_SPREAD_MODEL,
    run_sentinel2_unet_and_afi_segmentation,
)
from services.weather_terrain import fetch_fused_environment

# Load environment variables from .env
load_dotenv()

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SUPPORTED_SOURCES = {
    "VIIRS_SNPP_NRT": "VIIRS Suomi NPP (375m)",
    "VIIRS_NOAA20_NRT": "VIIRS NOAA-20 (375m)",
    "VIIRS_NOAA21_NRT": "VIIRS NOAA-21 (375m)",
    "MODIS_NRT": "MODIS Terra & Aqua (1km)",
}

REGION_BBOXES = {
    "world": "world",
    "north_america": "-168,15,-52,72",
    "south_america": "-82,-56,-34,13",
    "europe": "-11,35,40,71",
    "africa": "-18,-35,52,37",
    "asia": "40,5,150,75",
    "australia": "112,-44,155,-10",
    "india": "68,6,97,36",
}


class FireDetection(BaseModel):
    """Normalized satellite thermal-anomaly / active-fire detection record."""

    id: str = Field(..., description="Deterministic identifier for the detection")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    acq_date: str = Field(..., description="Acquisition date in YYYY-MM-DD (UTC)")
    acq_time: str = Field(..., description="Acquisition time in HHMM (UTC)")
    acq_datetime: str = Field(..., description="Combined UTC ISO-8601 timestamp")
    satellite: str = Field(..., description="Satellite identifier returned by dataset")
    instrument: Optional[str] = Field(None, description="Sensor instrument (e.g., VIIRS, MODIS)")
    confidence: Optional[str] = Field(None, description="Normalized confidence (low, nominal, high)")
    frp: Optional[float] = Field(None, description="Fire Radiative Power in MW")
    brightness: Optional[float] = Field(None, description="Brightness temperature in Kelvin (bright_ti4 or brightness)")
    bright_ti5: Optional[float] = Field(None, description="Channel I5/T31 brightness temperature in Kelvin")
    scan: Optional[float] = Field(None, description="Along-scan pixel size in km")
    track: Optional[float] = Field(None, description="Along-track pixel size in km")
    daynight: Optional[str] = Field(None, description="D for Daytime, N for Nighttime")


class FiresResponse(BaseModel):
    """Normalized response returned by GET /api/fires."""

    source: str = Field(default="NASA FIRMS")
    dataset: str = Field(..., description="FIRMS dataset identifier used for the query")
    region: str = Field(..., description="Geographic region queried")
    retrieved_at: str = Field(..., description="UTC ISO-8601 timestamp when data was fetched from NASA FIRMS")
    latest_observation_at: Optional[str] = Field(
        None, description="Latest observation UTC ISO-8601 timestamp among returned detections"
    )
    cached: bool = Field(default=False, description="True if served from in-memory cache")
    total_raw_detections: int = Field(..., description="Total detections observed in the dataset before limit")
    count: int = Field(..., description="Number of detections included in this response")
    fires: List[FireDetection]


# In-memory caches
_CACHE: Dict[str, Tuple[float, str, List[FireDetection]]] = {}
_INCIDENTS_CACHE: Dict[str, List[Dict[str, Any]]] = {}

app = FastAPI(
    title="WildfireWatch API",
    description=(
        "Near-Real-Time Disaster Perimeter & Spread Forecasting Engine. "
        "Integrates NASA FIRMS Thermal Telemetry, Sentinel-2 STAC Multispectral Metadata, "
        "and Open-Meteo Wind/Humidity/DEM Elevation APIs to extract active wildfire perimeters "
        "and forecast 6h & 12h spread."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_confidence(raw_conf: Optional[str]) -> Optional[str]:
    if raw_conf is None:
        return None
    val = raw_conf.strip().lower()
    if not val:
        return None
    if val in ("l", "low"):
        return "low"
    if val in ("n", "nominal"):
        return "nominal"
    if val in ("h", "high"):
        return "high"
    try:
        num = float(val)
        if num >= 80:
            return "high"
        if num >= 50:
            return "nominal"
        return "low"
    except ValueError:
        return val


def normalize_satellite(raw_sat: Optional[str], instrument: Optional[str]) -> str:
    if not raw_sat:
        return instrument or "Unknown"
    code = raw_sat.strip()
    mapping = {
        "N": "Suomi NPP (VIIRS)",
        "N20": "NOAA-20 (VIIRS)",
        "N21": "NOAA-21 (VIIRS)",
        "1": "NOAA-20 (VIIRS)",
        "2": "NOAA-21 (VIIRS)",
        "T": "Terra (MODIS)",
        "A": "Aqua (MODIS)",
    }
    return mapping.get(code, code)


def parse_optional_float(val: Optional[str]) -> Optional[float]:
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def build_utc_iso(acq_date: str, acq_time: str) -> Tuple[str, str]:
    padded_time = acq_time.strip().zfill(4)
    hh = padded_time[:2]
    mm = padded_time[2:4]
    iso_str = f"{acq_date.strip()}T{hh}:{mm}:00Z"
    return padded_time, iso_str


def parse_firms_csv(csv_text: str) -> List[FireDetection]:
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    detections: List[FireDetection] = []

    for idx, row in enumerate(reader):
        try:
            lat_raw = row.get("latitude")
            lon_raw = row.get("longitude")
            acq_date = row.get("acq_date")
            acq_time = row.get("acq_time")

            if not lat_raw or not lon_raw or not acq_date or not acq_time:
                continue

            lat = float(lat_raw)
            lon = float(lon_raw)
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue

            padded_time, acq_datetime = build_utc_iso(acq_date, acq_time)
            instrument = row.get("instrument")
            satellite = normalize_satellite(row.get("satellite"), instrument)
            confidence = normalize_confidence(row.get("confidence"))
            frp = parse_optional_float(row.get("frp"))
            brightness = parse_optional_float(row.get("bright_ti4") or row.get("brightness"))
            bright_ti5 = parse_optional_float(row.get("bright_ti5") or row.get("bright_t31"))
            scan = parse_optional_float(row.get("scan"))
            track = parse_optional_float(row.get("track"))
            daynight = row.get("daynight")

            det_id = f"{acq_date}_{padded_time}_{lat:.4f}_{lon:.4f}_{idx}"

            detections.append(
                FireDetection(
                    id=det_id,
                    latitude=lat,
                    longitude=lon,
                    acq_date=acq_date.strip(),
                    acq_time=padded_time,
                    acq_datetime=acq_datetime,
                    satellite=satellite,
                    instrument=instrument.strip() if instrument else None,
                    confidence=confidence,
                    frp=frp,
                    brightness=brightness,
                    bright_ti5=bright_ti5,
                    scan=scan,
                    track=track,
                    daynight=daynight.strip() if daynight else None,
                )
            )
        except (ValueError, TypeError):
            continue

    return detections


DEFAULT_FIRMS_MAP_KEY = "b1fbe81d2d44f435a5b8619919eae3c4"


async def _retrieve_firms_detections(
    dataset: str, region: str, days: int, refresh: bool
) -> Tuple[List[FireDetection], str, bool]:
    """Internal helper to fetch or return cached NASA FIRMS detections."""
    map_key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if not map_key or map_key == "your_firms_map_key_here":
        map_key = DEFAULT_FIRMS_MAP_KEY

    if dataset not in SUPPORTED_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset '{dataset}'. Supported: {list(SUPPORTED_SOURCES.keys())}",
        )

    area_coords = REGION_BBOXES.get(region.lower())
    if not area_coords:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported region '{region}'. Supported: {list(REGION_BBOXES.keys())}",
        )

    cache_ttl = int(os.getenv("FIRMS_CACHE_TTL_SECONDS", "300"))
    cache_key = f"{dataset}:{area_coords}:{days}"
    now_epoch = time.time()

    if not refresh and cache_key in _CACHE:
        cached_epoch, cached_iso, cached_items = _CACHE[cache_key]
        if now_epoch - cached_epoch < cache_ttl:
            return cached_items, cached_iso, True

    url = f"{FIRMS_BASE_URL}/{map_key}/{dataset}/{area_coords}/{days}"
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Request to NASA FIRMS timed out.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Network error contacting NASA FIRMS: {str(exc)}") from exc

    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="NASA FIRMS rate limit exceeded.")
    if response.status_code in (401, 403):
        raise HTTPException(status_code=502, detail="NASA FIRMS rejected the FIRMS_MAP_KEY.")
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"NASA FIRMS returned HTTP {response.status_code}.")

    body_text = response.text.strip()
    if "Invalid MAP_KEY" in body_text or "exceeded" in body_text.lower():
        raise HTTPException(status_code=502, detail=f"NASA FIRMS API error: {body_text[:200]}")

    all_detections = parse_firms_csv(body_text)
    retrieved_at_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _CACHE[cache_key] = (now_epoch, retrieved_at_iso, all_detections)
    return all_detections, retrieved_at_iso, False


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the interactive WildfireWatch Leaflet dashboard."""
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)


@app.get("/api/health")
async def health_check():
    """Check backend readiness and integrated external APIs."""
    map_key = os.getenv("FIRMS_MAP_KEY", DEFAULT_FIRMS_MAP_KEY).strip()
    has_valid_key = bool(map_key and map_key != "your_firms_map_key_here")
    return {
        "status": "ok",
        "service": "WildfireWatch Multi-API Engine",
        "version": "2.0.0",
        "firms_key_configured": has_valid_key,
        "external_apis_integrated": [
            "NASA FIRMS Area REST API (Active Fire & Thermal Anomalies)",
            "Open-Meteo Meteorological Forecast API (Hourly Wind Speed, Direction, Humidity, Temp)",
            "Open-Meteo SRTM/Copernicus DEM Elevation API (5-Point Topographical Slope)",
            "Element84 Earth Search STAC API (Sentinel-2 L2A Multispectral NIR/SWIR)",
        ],
        "supported_datasets": SUPPORTED_SOURCES,
        "supported_regions": list(REGION_BBOXES.keys()),
    }


@app.get("/api/fires", response_model=FiresResponse)
async def get_fires(
    dataset: str = Query(default=os.getenv("FIRMS_DEFAULT_SOURCE", "VIIRS_SNPP_NRT")),
    days: int = Query(default=1, ge=1, le=5),
    region: str = Query(default="world"),
    limit: int = Query(default=2500, ge=50, le=10000),
    min_confidence: Optional[str] = Query(default=None),
    refresh: bool = Query(default=False),
):
    """
    Milestone 1 Endpoint: Retrieve, validate, and normalize satellite active-fire detections from NASA FIRMS.
    """
    all_detections, retrieved_at_iso, is_cached = await _retrieve_firms_detections(
        dataset=dataset, region=region, days=days, refresh=refresh
    )

    filtered = all_detections
    if min_confidence:
        conf_filter = min_confidence.strip().lower()
        if conf_filter == "high":
            filtered = [d for d in filtered if d.confidence == "high"]
        elif conf_filter == "nominal":
            filtered = [d for d in filtered if d.confidence in ("nominal", "high")]

    total_raw = len(filtered)
    latest_obs = max((d.acq_datetime for d in filtered), default=None)

    sorted_fires = sorted(
        filtered,
        key=lambda d: (
            1 if d.confidence == "high" else (0 if d.confidence == "nominal" else -1),
            d.frp if d.frp is not None else -1.0,
            d.acq_datetime,
        ),
        reverse=True,
    )
    limited_fires = sorted_fires[:limit]

    return FiresResponse(
        source="NASA FIRMS",
        dataset=dataset,
        region=region.lower(),
        retrieved_at=retrieved_at_iso,
        latest_observation_at=latest_obs,
        cached=is_cached,
        total_raw_detections=total_raw,
        count=len(limited_fires),
        fires=limited_fires,
    )


@app.get("/api/incidents")
async def get_incidents(
    dataset: str = Query(default=os.getenv("FIRMS_DEFAULT_SOURCE", "VIIRS_SNPP_NRT")),
    days: int = Query(default=1, ge=1, le=5),
    region: str = Query(default="world"),
    cluster_radius_km: float = Query(default=3.2, ge=1.0, le=15.0),
    min_hotspots: int = Query(default=2, ge=1, le=20),
    max_incidents: int = Query(default=50, ge=5, le=150),
    refresh: bool = Query(default=False),
):
    """
    Stage 2 & Stage 3 Endpoint:
    Groups nearby NASA FIRMS hotspots into candidate Wildfire Incidents using spatial clustering,
    filters out isolated 1-dot low-FRP industrial/road noise, and extracts each incident's
    Current Disaster Perimeter Polygon (T+0h) as GeoJSON.
    """
    all_detections, retrieved_at_iso, is_cached = await _retrieve_firms_detections(
        dataset=dataset, region=region, days=days, refresh=refresh
    )

    # Filter out low-confidence false alarms before incident clustering
    credible = [d.model_dump() for d in all_detections if d.confidence in ("nominal", "high")]

    incidents = cluster_hotspots_into_incidents(
        detections=credible,
        cluster_radius_km=cluster_radius_km,
        min_hotspots=min_hotspots,
        max_incidents=max_incidents,
    )

    cache_scope = f"{dataset}:{region.lower()}:{days}"
    _INCIDENTS_CACHE[cache_scope] = incidents
    for inc in incidents:
        _INCIDENTS_CACHE[inc["incident_id"]] = [inc]

    return {
        "source": "NASA FIRMS + WildfireWatch Spatial Clustering Engine",
        "dataset": dataset,
        "region": region.lower(),
        "retrieved_at": retrieved_at_iso,
        "cached": is_cached,
        "clustering_parameters": {
            "eps_radius_km": cluster_radius_km,
            "min_hotspots": min_hotspots,
            "noise_filtering": "Excludes low-confidence & isolated low-FRP non-wildfire anomalies",
        },
        "incident_count": len(incidents),
        "incidents": incidents,
    }


@app.get("/api/incidents/{incident_id}/forecast")
async def get_incident_forecast(
    incident_id: str,
    dataset: str = Query(default=os.getenv("FIRMS_DEFAULT_SOURCE", "VIIRS_SNPP_NRT")),
    region: str = Query(default="world"),
    days: int = Query(default=1, ge=1, le=5),
):
    """
    Stage 4, 5 & 6 Endpoint:
    For a selected Wildfire Incident:
    1. Fetches live hourly Wind Speed, Wind Direction, Relative Humidity & Temperature from Open-Meteo API
    2. Fetches 5-point Digital Elevation Model (DEM) terrain slope gradient from Open-Meteo Elevation API
    3. Queries Sentinel-2 L2A Multispectral STAC API for latest NIR/SWIR scene metadata
    4. Runs the 6h & 12h Elliptical Wavelet / Rothermel Spread Forecast Engine
    5. Returns T+0h Observed Perimeter + T+6h & T+12h Forecast Polygons + Compression Telemetry
    """
    target_incident: Optional[Dict[str, Any]] = None
    if incident_id in _INCIDENTS_CACHE and _INCIDENTS_CACHE[incident_id]:
        target_incident = _INCIDENTS_CACHE[incident_id][0]
    else:
        # Re-cluster if cache was cold
        inc_resp = await get_incidents(dataset=dataset, days=days, region=region)
        for inc in inc_resp["incidents"]:
            if inc["incident_id"] == incident_id:
                target_incident = inc
                break

    if not target_incident:
        raise HTTPException(
            status_code=404,
            detail=f"Incident '{incident_id}' not found. Call GET /api/incidents first.",
        )

    lat = target_incident["centroid"]["latitude"]
    lon = target_incident["centroid"]["longitude"]

    # Fetch live Weather + Topographical DEM Slope + Sentinel-2 STAC metadata in parallel
    env_data, stac_metadata = await asyncio.gather(
        fetch_fused_environment(lat, lon),
        query_sentinel2_stac_metadata(lat, lon),
    )

    forecast_bundle = simulate_spread_polygons(
        incident=target_incident,
        weather_data=env_data["weather"],
        terrain_data=env_data["terrain"],
    )

    # Run Sentinel-2 3-Band (B12 SWIR2, B11 SWIR1, B8A NIR) U-Net + Liu/Murphy AFI Segmentation
    unet_segmentation = run_sentinel2_unet_and_afi_segmentation(target_incident)

    # Run 100-Tree Random Forest Spread Ensemble & Feature Importance attribution
    w_curr = env_data["weather"]["current"]
    t_curr = env_data["terrain"]
    mean_frp = target_incident["total_frp_mw"] / max(1, target_incident["hotspot_count"])
    ml_ensemble = RF_SPREAD_MODEL.predict_with_importance(
        wind_speed_kmh=float(w_curr["wind_speed_kmh"]),
        wind_gusts_kmh=float(w_curr["wind_gusts_kmh"]),
        relative_humidity_pct=float(w_curr["relative_humidity_pct"]),
        temperature_c=float(w_curr["temperature_c"]),
        slope_degrees=float(t_curr["slope_degrees"]),
        mean_frp_mw=float(mean_frp),
        max_brightness_k=float(target_incident["max_brightness_k"]),
    )

    return {
        "incident": {
            "incident_id": target_incident["incident_id"],
            "name": target_incident["name"],
            "severity": target_incident["severity"],
            "centroid": target_incident["centroid"],
            "hotspot_count": target_incident["hotspot_count"],
            "total_frp_mw": target_incident["total_frp_mw"],
            "max_brightness_k": target_incident["max_brightness_k"],
            "latest_observation_at": target_incident["latest_observation_at"],
            "extraction_method": target_incident["extraction_method"],
        },
        "environment": env_data,
        "multispectral_stac": stac_metadata,
        "unet_segmentation": unet_segmentation,
        "ml_spread_ensemble": ml_ensemble,
        "forecast": forecast_bundle,
    }


@app.get("/api/incidents/{incident_id}/export")
async def export_tactical_geojson(
    incident_id: str,
    dataset: str = Query(default=os.getenv("FIRMS_DEFAULT_SOURCE", "VIIRS_SNPP_NRT")),
    region: str = Query(default="world"),
    days: int = Query(default=1, ge=1, le=5),
):
    """
    Evaluation Criterion #3 Endpoint:
    Exports an ultra-lightweight, RFC 7946 compliant GeoJSON FeatureCollection containing
    the Observed Perimeter (T+0h), +6h Spread Forecast, +12h Spread Forecast, and
    embedded wind/terrain telemetry for immediate field responder deployment.
    """
    analysis = await get_incident_forecast(
        incident_id=incident_id, dataset=dataset, region=region, days=days
    )
    geojson_payload = analysis["forecast"]["geojson_feature_collection"]
    geojson_payload["metadata"]["environment_summary"] = {
        "wind_speed_kmh": analysis["environment"]["weather"]["current"]["wind_speed_kmh"],
        "wind_from_deg": analysis["environment"]["weather"]["current"]["wind_from_deg"],
        "wind_toward_cardinal": analysis["environment"]["weather"]["current"]["wind_toward_cardinal"],
        "relative_humidity_pct": analysis["environment"]["weather"]["current"]["relative_humidity_pct"],
        "temperature_c": analysis["environment"]["weather"]["current"]["temperature_c"],
        "slope_degrees": analysis["environment"]["terrain"]["slope_degrees"],
        "uphill_cardinal": analysis["environment"]["terrain"]["uphill_cardinal"],
    }
    geojson_payload["metadata"]["pipeline_efficiency"] = analysis["forecast"]["pipeline_efficiency_metrics"]

    return JSONResponse(
        content=geojson_payload,
        headers={
            "Content-Disposition": f'attachment; filename="{incident_id}_tactical_forecast.geojson"'
        },
    )
