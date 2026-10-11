# WildfireWatch — Near-Real-Time Disaster Perimeter & Spread Forecasting

**WildfireWatch** is a live geospatial intelligence web application and FastAPI engine built for the **Near-Real-Time Disaster Perimeter & Spread Forecasting** problem statement.

It ingests live satellite thermal anomalies from **NASA FIRMS**, clusters scattered detections into active wildfire incidents (`DBSCAN`), verifies active burning fronts using **Sentinel-2 Short-Wave Infrared (`B12/B11/B8A`) U-Net / Active Fire Index** segmentation, fuses live meteorological and terrain slope data from **Open-Meteo**, and predicts **`+6h` and `+12h` wildfire spread polygons** using a **100-Tree Random Forest Spread Ensemble + Rothermel Elliptical Wavelet model**.

---

## Quick Start (How to Run after Cloning from GitHub)

> **No API Key Setup Required!**
> - **NASA FIRMS API Key:** Pre-configured with a built-in fallback key in `backend/main.py` (you can also override it via `FIRMS_MAP_KEY` in `backend/.env`).
> - **Open-Meteo Weather API, Open-Meteo Elevation (DEM) API, and Element84 Sentinel-2 STAC API:** 100% keyless public APIs.

### Option 1: One-Click Run on Windows
Simply double-click **`run.bat`** in the project root, or run:
```powershell
.\run.bat
```
Then open **`http://127.0.0.1:8000/`** in your browser.

### Option 2: Standard Terminal Commands (Windows / macOS / Linux)
**Prerequisites:** Python 3.9+ installed.

```bash
# 1. Enter the backend folder
cd backend

# 2. (Optional) Create & activate a virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server (serves both the GIS Frontend and REST API)
uvicorn main:app --host 127.0.0.1 --port 8000
```

Once started:
- **Interactive GIS Web App:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Live APIs Integrated

| API / Data Source | Purpose in Pipeline | Auth Required |
| :--- | :--- | :--- |
| **1. NASA FIRMS Area REST API** | Live `VIIRS S-NPP` ($375\text{ m}$) & `MODIS` ($1\text{ km}$) thermal anomalies (`lat`, `lon`, `FRP` in MW, brightness temp, UTC timestamp) | Built-in `FIRMS_MAP_KEY` included |
| **2. Element84 Sentinel-2 L2A STAC API** | Queries latest Sentinel-2 L2A satellite pass & cloud cover over the incident bounding box | Keyless Public API |
| **3. Open-Meteo Weather Forecast API** | Fetches live hourly `10m` Wind Speed ($\text{km/h}$), Wind Direction ($^\circ$), `2m` Relative Humidity ($\%$), and Air Temperature ($^\circ\text{C}$) | Keyless Public API |
| **4. Open-Meteo DEM Elevation API** | Samples 5 elevation points ($1\text{ km}$ N, S, E, W, and Center) to compute terrain slope ($^\circ$) and uphill aspect | Keyless Public API |

---

## Machine Learning & Physics Forecasting Pipeline

1. **Stage 1 — Unsupervised DBSCAN Spatial Clustering (`backend/services/perimeter_engine.py`):**
   - Uses Haversine geodesic distance ($\varepsilon = 15\text{ km}$) to automatically group raw satellite thermal hotspots into discrete **Wildfire Incidents** and extracts the **`T+0h` active perimeter polygon** in hectares.
2. **Stage 2 — Sentinel-2 SWIR U-Net & Active Fire Index (`backend/services/unet_and_ml_engine.py`):**
   - Incorporates transfer-learning U-Net segmentation concepts from Landsat-8 to Sentinel-2 Short-Wave Infrared bands (`B12` $2190\text{ nm}$, `B11` $1610\text{ nm}$, `B8A` $865\text{ nm}$) to penetrate smoke and isolate active burning fronts.
3. **Stage 3 — 100-Tree Random Forest Spread Ensemble (`backend/services/unet_and_ml_engine.py`):**
   - Fuses 6 live features (*Wind Speed, Relative Humidity, Air Temperature, Terrain Slope, Fire Radiative Power, and SWIR Active Fire Index*) across 100 decision trees to predict the forward **Rate of Spread ($\text{m/min}$)** and compute the **Forecast Confidence Score ($\%$)**.
4. **Stage 4 — Rothermel Elliptical Wavelet Propagation & Hindcast Validation (`backend/services/spread_forecaster.py`):**
   - Projects the Random Forest spread rate along the wind and uphill slope vector to generate **`+6h` (Orange)** and **`+12h` (Yellow)** forecast polygons, and computes a **Hindcast Validation Score ($\%$)** from multi-pass satellite progression.

---

## REST API Endpoints

- `GET /` — Serves the interactive Leaflet GIS web interface (`backend/static/index.html`).
- `GET /api/health` — Returns system health, configured APIs, and supported satellite datasets/regions.
- `GET /api/fires?region=world&days=1` — Returns normalized live NASA FIRMS thermal detections.
- `GET /api/incidents?region=world&days=1` — Returns DBSCAN-clustered wildfire incidents with `T+0h` GeoJSON polygons and area in hectares.
- `GET /api/incidents/{incident_id}/forecast?region=world&days=1` — Runs live Open-Meteo weather/DEM fusion, Sentinel-2 U-Net/AFI segmentation, Random Forest spread rate prediction, and returns `+6h` & `+12h` forecast polygons.
- `GET /api/incidents/{incident_id}/export?region=world&days=1` — Downloads a standalone `RFC 7946` `.geojson` FeatureCollection (`T+0h`, `+6h`, `+12h` perimeters) for GIS tools.
