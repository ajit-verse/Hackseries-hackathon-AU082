"""
Stage 2 & Stage 3: Spatial Incident Clustering & Fire Perimeter Extraction Engine.
Groups individual satellite thermal hotspot detections into candidate wildfire incidents
and extracts a geographic perimeter polygon (GeoJSON) + queries Sentinel-2 STAC API.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import httpx


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def offset_lat_lon(lat: float, lon: float, north_km: float, east_km: float) -> Tuple[float, float]:
    """Offset a (lat, lon) coordinate by north_km and east_km."""
    d_lat = (north_km / EARTH_RADIUS_KM) * (180.0 / math.pi)
    cos_lat = max(0.05, math.cos(math.radians(lat)))
    d_lon = (east_km / (EARTH_RADIUS_KM * cos_lat)) * (180.0 / math.pi)
    return round(lat + d_lat, 5), round(lon + d_lon, 5)


def polygon_area_hectares(coords_lon_lat: List[List[float]], ref_lat: float) -> float:
    """Compute approximate area of a closed [lon, lat] polygon in hectares using local equirectangular projection."""
    if len(coords_lon_lat) < 4:
        return 0.0
    cos_lat = math.cos(math.radians(ref_lat))
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos_lat

    pts_km = [
        (pt[0] * km_per_deg_lon, pt[1] * km_per_deg_lat)
        for pt in coords_lon_lat
    ]
    area_km2 = 0.0
    for i in range(len(pts_km) - 1):
        x1, y1 = pts_km[i]
        x2, y2 = pts_km[i + 1]
        area_km2 += (x1 * y2) - (x2 * y1)
    area_km2 = abs(area_km2) * 0.5
    return round(area_km2 * 100.0, 1)  # 1 km^2 = 100 hectares


def extract_perimeter_polygon(hotspots: List[Dict[str, Any]], n_sectors: int = 24) -> Tuple[List[List[float]], float]:
    """
    Extract a smooth radial concave/alpha-hull boundary around a cluster of thermal hotspots.
    Each hotspot is expanded by its actual satellite pixel footprint (scan x track in km,
    plus thermal bloom margin) so the boundary reflects sensor footprint geometry.
    Returns (closed GeoJSON ring [[lon, lat], ...], area_hectares).
    """
    n = len(hotspots)
    center_lat = sum(h["latitude"] for h in hotspots) / n
    center_lon = sum(h["longitude"] for h in hotspots) / n

    cos_lat = max(0.05, math.cos(math.radians(center_lat)))
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos_lat

    # Convert hotspot centers to local (east_km, north_km) relative to cluster centroid
    local_pts: List[Tuple[float, float, float]] = []
    for h in hotspots:
        east_km = (h["longitude"] - center_lon) * km_per_deg_lon
        north_km = (h["latitude"] - center_lat) * km_per_deg_lat
        scan_km = h.get("scan") or 0.4
        track_km = h.get("track") or 0.4
        frp = h.get("frp") or 5.0
        # Footprint radius from sensor pixel half-diagonal + radiative intensity margin
        pixel_radius_km = max(0.35, 0.65 * math.hypot(scan_km, track_km) + min(0.45, math.sqrt(max(0.0, frp)) * 0.03))
        local_pts.append((east_km, north_km, pixel_radius_km))

    # Sample boundary envelope across angular sectors (0 to 2*pi)
    sector_radii: List[float] = []
    for s in range(n_sectors):
        theta = (2.0 * math.pi * s) / n_sectors
        ux = math.cos(theta)  # East unit component
        uy = math.sin(theta)  # North unit component

        max_proj = 0.35
        for px, py, pr in local_pts:
            # Projection along ray theta and perpendicular distance to ray
            along = px * ux + py * uy
            perp = abs(-px * uy + py * ux)
            # Contrib of this hotspot's buffered footprint along angle theta
            if perp <= pr * 1.8:
                reach = along + math.sqrt(max(0.04, (pr * 1.35) ** 2 - (perp * 0.65) ** 2))
            else:
                # Smooth envelope support along direction theta
                reach = along * 0.85 + pr
            if reach > max_proj:
                max_proj = reach
        sector_radii.append(max(0.35, max_proj))

    # 3-point circular moving-average smoothing to produce a realistic organic fire perimeter
    smoothed_radii: List[float] = []
    for s in range(n_sectors):
        prev_r = sector_radii[(s - 1) % n_sectors]
        curr_r = sector_radii[s]
        next_r = sector_radii[(s + 1) % n_sectors]
        smoothed_radii.append(0.25 * prev_r + 0.50 * curr_r + 0.25 * next_r)

    ring_lon_lat: List[List[float]] = []
    for s in range(n_sectors):
        theta = (2.0 * math.pi * s) / n_sectors
        r_km = smoothed_radii[s]
        east_km = r_km * math.cos(theta)
        north_km = r_km * math.sin(theta)
        lat_pt, lon_pt = offset_lat_lon(center_lat, center_lon, north_km, east_km)
        ring_lon_lat.append([lon_pt, lat_pt])

    # Close GeoJSON LinearRing
    ring_lon_lat.append(ring_lon_lat[0])
    area_ha = polygon_area_hectares(ring_lon_lat, center_lat)
    return ring_lon_lat, area_ha


def cluster_hotspots_into_incidents(
    detections: List[Dict[str, Any]],
    cluster_radius_km: float = 3.2,
    min_hotspots: int = 2,
    max_incidents: int = 60,
) -> List[Dict[str, Any]]:
    """
    Group nearby satellite thermal detections into candidate Wildfire Incidents
    using spatial grid-accelerated connected-components (DBSCAN-style epsilon neighborhood).
    Filters out isolated 1-dot low-FRP industrial/roadside thermal noise.
    """
    if not detections:
        return []

    # Spatial hash grid (~0.03 deg cells ~ 3.3 km) for O(N) neighbor lookup
    cell_size = 0.03
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, d in enumerate(detections):
        gx = int(math.floor(d["latitude"] / cell_size))
        gy = int(math.floor(d["longitude"] / cell_size))
        grid.setdefault((gx, gy), []).append(idx)

    visited = [False] * len(detections)
    clusters: List[List[Dict[str, Any]]] = []

    for i in range(len(detections)):
        if visited[i]:
            continue
        visited[i] = True
        comp_indices = [i]
        queue = [i]

        while queue:
            curr = queue.pop()
            clat = detections[curr]["latitude"]
            clon = detections[curr]["longitude"]
            gx = int(math.floor(clat / cell_size))
            gy = int(math.floor(clon / cell_size))

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for nb in grid.get((gx + dx, gy + dy), []):
                        if not visited[nb]:
                            dist = haversine_km(
                                clat, clon,
                                detections[nb]["latitude"], detections[nb]["longitude"]
                            )
                            if dist <= cluster_radius_km:
                                visited[nb] = True
                                comp_indices.append(nb)
                                queue.append(nb)

        cluster_pts = [detections[idx] for idx in comp_indices]
        total_frp = sum(p.get("frp") or 0.0 for p in cluster_pts)
        max_frp = max((p.get("frp") or 0.0) for p in cluster_pts)

        # Keep clusters with >= min_hotspots OR single extreme high-confidence wildfire (FRP >= 35 MW)
        if len(cluster_pts) >= min_hotspots or (
            max_frp >= 35.0 and cluster_pts[0].get("confidence") == "high"
        ):
            clusters.append(cluster_pts)

    # Sort candidate incidents by total radiative power and hotspot count
    clusters.sort(
        key=lambda c: (
            len(c) >= 3,
            sum(p.get("frp") or 0.0 for p in c),
            len(c),
        ),
        reverse=True,
    )

    incidents: List[Dict[str, Any]] = []
    for rank, cluster in enumerate(clusters[:max_incidents], start=1):
        n = len(cluster)
        centroid_lat = round(sum(p["latitude"] for p in cluster) / n, 4)
        centroid_lon = round(sum(p["longitude"] for p in cluster) / n, 4)
        total_frp = round(sum(p.get("frp") or 0.0 for p in cluster), 1)
        max_frp = round(max((p.get("frp") or 0.0) for p in cluster), 1)
        max_brightness = round(max((p.get("brightness") or 300.0) for p in cluster), 1)
        latest_obs = max(p["acq_datetime"] for p in cluster)
        high_conf_count = sum(1 for p in cluster if p.get("confidence") == "high")

        ring_coords, area_ha = extract_perimeter_polygon(cluster, n_sectors=24)

        inc_id = f"INC-{rank:03d}-{abs(int(centroid_lat * 100)):04d}-{abs(int(centroid_lon * 100)):05d}"

        if total_frp >= 150 or n >= 12:
            severity = "EXTREME"
        elif total_frp >= 50 or n >= 5:
            severity = "HIGH"
        else:
            severity = "MODERATE"

        incidents.append(
            {
                "incident_id": inc_id,
                "name": f"Wildfire Incident #{rank} ({centroid_lat:+.2f}°, {centroid_lon:+.2f}°)",
                "severity": severity,
                "centroid": {"latitude": centroid_lat, "longitude": centroid_lon},
                "hotspot_count": n,
                "high_confidence_count": high_conf_count,
                "total_frp_mw": total_frp,
                "max_frp_mw": max_frp,
                "max_brightness_k": max_brightness,
                "latest_observation_at": latest_obs,
                "area_hectares": area_ha,
                "extraction_method": (
                    "DBSCAN Spatial Clustering (r=3.2km) + Sensor-Footprint Alpha Envelope (VIIRS 375m)"
                ),
                "perimeter_geojson": {
                    "type": "Feature",
                    "properties": {
                        "incident_id": inc_id,
                        "layer_type": "observed_perimeter_t0",
                        "label": "Current Extracted Fire Perimeter (T+0h)",
                        "area_hectares": area_ha,
                        "hotspot_count": n,
                        "total_frp_mw": total_frp,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ring_coords],
                    },
                },
                "hotspots": cluster[:40],  # Include constituent hotspots for inspection
            }
        )

    return incidents


async def query_sentinel2_stac_metadata(lat: float, lon: float) -> Dict[str, Any]:
    """
    Query the public Element84 Earth Search STAC API (Sentinel-2 L2A Multispectral Collection)
    for the bounding box around the wildfire incident to retrieve NIR (B08) & SWIR (B12) scene metadata.
    """
    bbox = [round(lon - 0.08, 4), round(lat - 0.08, 4), round(lon + 0.08, 4), round(lat + 0.08, 4)]
    stac_url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "limit": 1,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(stac_url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                features = data.get("features", [])
                if features:
                    feat = features[0]
                    props = feat.get("properties", {})
                    assets = feat.get("assets", {})
                    return {
                        "available": True,
                        "stac_endpoint": stac_url,
                        "collection": "sentinel-2-l2a (Multispectral 10m/20m COG)",
                        "scene_id": feat.get("id"),
                        "captured_at": props.get("datetime"),
                        "cloud_cover_pct": round(float(props.get("eo:cloud_cover", 0.0)), 1),
                        "platform": props.get("platform", "sentinel-2"),
                        "spectral_bands_used": {
                            "NIR_Band_8A": assets.get("nir08", {}).get("href") or assets.get("nir", {}).get("href"),
                            "SWIR_Band_12": assets.get("swir22", {}).get("href"),
                            "Visual_Thumbnail": assets.get("thumbnail", {}).get("href"),
                        },
                        "nbr_index_formula": "NBR = (NIR_B8A - SWIR_B12) / (NIR_B8A + SWIR_B12)",
                    }
    except Exception as exc:
        return {
            "available": False,
            "stac_endpoint": stac_url,
            "note": f"STAC lookup skipped or timed out ({str(exc)[:80]})",
        }

    return {
        "available": False,
        "stac_endpoint": stac_url,
        "note": "No recent Sentinel-2 L2A scene returned for bounding box",
    }
