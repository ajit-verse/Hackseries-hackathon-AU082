import React, { useEffect, useState, useMemo } from 'react';
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap,
} from 'react-leaflet';
import {
  Flame,
  RefreshCw,
  Satellite,
  Clock,
  AlertTriangle,
  Radio,
  Globe,
  MapPin,
  Thermometer,
  Zap,
  Info,
  CheckCircle2,
} from 'lucide-react';

const REGION_VIEWS = {
  world: { center: [20, 10], zoom: 2, label: 'Global (World)' },
  north_america: { center: [40, -100], zoom: 4, label: 'North America' },
  south_america: { center: [-15, -60], zoom: 4, label: 'South America' },
  europe: { center: [50, 15], zoom: 4, label: 'Europe' },
  africa: { center: [2, 20], zoom: 4, label: 'Africa' },
  asia: { center: [30, 95], zoom: 4, label: 'Asia' },
  australia: { center: [-25, 134], zoom: 4, label: 'Australia' },
  india: { center: [22, 79], zoom: 5, label: 'India' },
};

const DATASETS = [
  { id: 'VIIRS_SNPP_NRT', label: 'VIIRS Suomi NPP (375m)' },
  { id: 'VIIRS_NOAA20_NRT', label: 'VIIRS NOAA-20 (375m)' },
  { id: 'MODIS_NRT', label: 'MODIS Terra & Aqua (1km)' },
];

function MapUpdater({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center && zoom) {
      map.flyTo(center, zoom, { duration: 1.2 });
    }
  }, [center, zoom, map]);
  return null;
}

function getHotspotColor(confidence, frp) {
  if (confidence === 'high' || (frp && frp >= 50)) {
    return { stroke: '#ef4444', fill: '#dc2626' }; // High intensity red
  }
  if (confidence === 'low') {
    return { stroke: '#facc15', fill: '#eab308' }; // Low confidence yellow
  }
  return { stroke: '#fb923c', fill: '#f97316' }; // Nominal orange
}

function getHotspotRadius(frp) {
  if (!frp || frp <= 0) return 5;
  if (frp > 100) return 9;
  if (frp > 30) return 7;
  return 5;
}

function formatUtcTimestamp(isoStr) {
  if (!isoStr) return 'N/A';
  try {
    const d = new Date(isoStr);
    return d.toISOString().replace('T', ' ').replace('.000Z', ' UTC').replace('Z', ' UTC');
  } catch {
    return isoStr;
  }
}

export default function App() {
  const [dataset, setDataset] = useState('VIIRS_SNPP_NRT');
  const [region, setRegion] = useState('world');
  const [days, setDays] = useState(1);
  const [limit, setLimit] = useState(2000);

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedFire, setSelectedFire] = useState(null);
  const [mapView, setMapView] = useState(REGION_VIEWS.world);

  const fetchFires = async (forceRefresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        dataset,
        region,
        days: String(days),
        limit: String(limit),
        refresh: String(forceRefresh),
      });
      const response = await fetch(`/api/fires?${params.toString()}`);
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail || `HTTP ${response.status} error from backend`);
      }

      setData(payload);
      if (payload.fires && payload.fires.length > 0 && !selectedFire) {
        setSelectedFire(payload.fires[0]);
      }
    } catch (err) {
      setError(err.message || 'Failed to retrieve wildfire detections from backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setMapView(REGION_VIEWS[region] || REGION_VIEWS.world);
    fetchFires(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, region, days, limit]);

  const handleSelectDetection = (fire) => {
    setSelectedFire(fire);
    setMapView({
      center: [fire.latitude, fire.longitude],
      zoom: 8,
    });
  };

  const recentDetections = useMemo(() => {
    if (!data?.fires) return [];
    return data.fires.slice(0, 60);
  }, [data]);

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Top Command Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur px-5 py-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-orange-500/15 border border-orange-500/40 flex items-center justify-center text-orange-400">
            <Flame className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-white">
                WildfireWatch
              </h1>
              <span className="text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                <Radio className="w-3 h-3 animate-pulse" />
                Live Telemetry
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Near-Real-Time Satellite Thermal Anomaly &amp; Active Fire Monitoring
            </p>
          </div>
        </div>

        {/* Filter & Refresh Controls */}
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex items-center gap-1.5 bg-slate-800/80 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs">
            <Satellite className="w-3.5 h-3.5 text-orange-400" />
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="bg-transparent text-slate-100 focus:outline-none cursor-pointer"
              disabled={loading}
            >
              {DATASETS.map((ds) => (
                <option key={ds.id} value={ds.id} className="bg-slate-900">
                  {ds.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-800/80 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs">
            <Globe className="w-3.5 h-3.5 text-sky-400" />
            <select
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              className="bg-transparent text-slate-100 focus:outline-none cursor-pointer"
              disabled={loading}
            >
              {Object.entries(REGION_VIEWS).map(([key, cfg]) => (
                <option key={key} value={key} className="bg-slate-900">
                  {cfg.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-800/80 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs">
            <Clock className="w-3.5 h-3.5 text-amber-400" />
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="bg-transparent text-slate-100 focus:outline-none cursor-pointer"
              disabled={loading}
            >
              <option value={1} className="bg-slate-900">Last 24 Hours (1 Day)</option>
              <option value={2} className="bg-slate-900">Last 48 Hours (2 Days)</option>
              <option value={3} className="bg-slate-900">Last 72 Hours (3 Days)</option>
            </select>
          </div>

          <button
            onClick={() => fetchFires(true)}
            disabled={loading}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-white text-xs font-semibold transition shadow-sm"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Fetching...' : 'Refresh Live Data'}
          </button>
        </div>
      </header>

      {/* Telemetry Metrics Strip */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 px-5 py-3 bg-slate-900/40 border-b border-slate-800/80">
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
              Returned Hotspot Detections
            </p>
            <p className="text-2xl font-bold text-white mt-0.5">
              {data ? data.count.toLocaleString() : '—'}
              {data && data.total_raw_detections > data.count && (
                <span className="text-xs font-normal text-slate-400 ml-1.5">
                  of {data.total_raw_detections.toLocaleString()} total in region
                </span>
              )}
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Individual thermal anomalies (multiple points may belong to one fire)
            </p>
          </div>
          <Flame className="w-8 h-8 text-orange-500/70 shrink-0" />
        </div>

        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
              Latest Satellite Observation
            </p>
            <p className="text-sm font-semibold text-amber-300 mt-1">
              {data?.latest_observation_at
                ? formatUtcTimestamp(data.latest_observation_at)
                : '—'}
            </p>
            <p className="text-[11px] text-slate-500 mt-1">
              Newest pass timestamp in returned dataset
            </p>
          </div>
          <Clock className="w-8 h-8 text-amber-400/70 shrink-0" />
        </div>

        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
              Data Source &amp; Sensor
            </p>
            <p className="text-sm font-semibold text-sky-300 mt-1">
              {data ? `${data.source} (${data.dataset})` : 'NASA FIRMS'}
            </p>
            <p className="text-[11px] text-slate-500 mt-1">
              Official NASA EOSDIS Active Fire stream
            </p>
          </div>
          <Satellite className="w-8 h-8 text-sky-400/70 shrink-0" />
        </div>

        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
              Feed Freshness Status
            </p>
            <p className="text-sm font-semibold text-emerald-300 mt-1">
              {data?.retrieved_at ? formatUtcTimestamp(data.retrieved_at) : '—'}
            </p>
            <p className="text-[11px] text-slate-500 mt-1">
              {data
                ? data.cached
                  ? 'Served from backend cache (5-min TTL)'
                  : 'Freshly retrieved from NASA FIRMS API'
                : 'Connecting to backend...'}
            </p>
          </div>
          <CheckCircle2 className="w-8 h-8 text-emerald-400/70 shrink-0" />
        </div>
      </section>

      {/* Error State Banner */}
      {error && (
        <div className="mx-5 mt-3 p-3.5 rounded-xl bg-red-950/80 border border-red-500/50 text-red-200 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 text-sm">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
            <span>
              <strong>Data Retrieval Error:</strong> {error}
            </span>
          </div>
          <button
            onClick={() => fetchFires(true)}
            className="px-3 py-1 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-400/40 text-xs font-medium shrink-0"
          >
            Retry
          </button>
        </div>
      )}

      {/* Main Workspace: Interactive World Map + Recent Detections Panel */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 p-5 min-h-[600px]">
        {/* Left/Center: Interactive World Map */}
        <div className="lg:col-span-8 xl:col-span-9 relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-900 flex flex-col min-h-[520px]">
          {/* Loading Overlay */}
          {loading && (
            <div className="absolute inset-0 z-[1000] bg-slate-950/75 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
              <RefreshCw className="w-8 h-8 text-orange-500 animate-spin" />
              <p className="text-sm font-medium text-slate-200">
                Retrieving live satellite thermal detections from NASA FIRMS...
              </p>
            </div>
          )}

          {/* Empty State Overlay */}
          {!loading && !error && data && data.fires.length === 0 && (
            <div className="absolute inset-0 z-[900] bg-slate-950/70 flex flex-col items-center justify-center p-6 text-center">
              <Flame className="w-10 h-10 text-slate-500 mb-2" />
              <h3 className="text-base font-semibold text-white">
                No Active Hotspot Detections Returned
              </h3>
              <p className="text-xs text-slate-400 max-w-md mt-1">
                No thermal anomalies matched the selected region and time range. Try selecting Global (World) or increasing the time window.
              </p>
            </div>
          )}

          <MapContainer
            center={mapView.center}
            zoom={mapView.zoom}
            className="w-full h-full flex-1 z-10"
            preferCanvas={true}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <MapUpdater center={mapView.center} zoom={mapView.zoom} />

            {data?.fires?.map((fire) => {
              const colors = getHotspotColor(fire.confidence, fire.frp);
              const radius = getHotspotRadius(fire.frp);
              return (
                <CircleMarker
                  key={fire.id}
                  center={[fire.latitude, fire.longitude]}
                  radius={radius}
                  pathOptions={{
                    color: colors.stroke,
                    fillColor: colors.fill,
                    fillOpacity: 0.8,
                    weight: 1.5,
                  }}
                  eventHandlers={{
                    click: () => setSelectedFire(fire),
                  }}
                >
                  <Popup>
                    <div className="text-xs space-y-1.5 min-w-[210px]">
                      <div className="flex items-center justify-between border-b border-slate-700 pb-1.5">
                        <span className="font-bold text-orange-400 flex items-center gap-1">
                          <Flame className="w-3.5 h-3.5" /> Satellite Hotspot
                        </span>
                        <span className="uppercase px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-200">
                          {fire.confidence || 'nominal'} conf
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-1 text-slate-300 pt-1">
                        <span className="text-slate-400">Coordinates:</span>
                        <span className="font-mono text-right">
                          {fire.latitude.toFixed(4)}°, {fire.longitude.toFixed(4)}°
                        </span>
                        <span className="text-slate-400">Observed (UTC):</span>
                        <span className="text-right">
                          {fire.acq_date} {fire.acq_time.slice(0, 2)}:{fire.acq_time.slice(2, 4)}
                        </span>
                        <span className="text-slate-400">Satellite:</span>
                        <span className="text-right">{fire.satellite}</span>
                        <span className="text-slate-400">Radiative Power:</span>
                        <span className="text-right font-semibold text-amber-300">
                          {fire.frp !== null ? `${fire.frp} MW` : 'N/A'}
                        </span>
                        <span className="text-slate-400">Brightness (I4):</span>
                        <span className="text-right">
                          {fire.brightness !== null ? `${fire.brightness} K` : 'N/A'}
                        </span>
                        <span className="text-slate-400">Pixel Footprint:</span>
                        <span className="text-right">
                          {fire.scan && fire.track ? `${fire.scan}×${fire.track} km` : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>

          {/* Map Legend (Explicitly noting hotspots vs fire boundaries per requirements) */}
          <div className="absolute bottom-4 left-4 z-[500] bg-slate-950/90 backdrop-blur border border-slate-800 rounded-xl p-3 shadow-xl max-w-xs text-xs">
            <div className="font-semibold text-slate-200 mb-2 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-orange-400" />
              Hotspot Legend &amp; Notice
            </div>
            <div className="space-y-1.5 text-[11px] text-slate-300">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-red-600 border border-red-400 inline-block shrink-0" />
                <span>High Confidence / High FRP (&ge; 50 MW)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-orange-500 border border-orange-400 inline-block shrink-0" />
                <span>Nominal Confidence Thermal Anomaly</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-yellow-500 border border-yellow-300 inline-block shrink-0" />
                <span>Low Confidence Thermal Anomaly</span>
              </div>
            </div>
            <p className="mt-2 pt-2 border-t border-slate-800 text-[10px] text-slate-400 leading-relaxed">
              <strong>Note:</strong> Markers represent satellite-detected thermal hotspots (pixel centers), <em>not</em> exact wildfire boundary polygons.
            </p>
          </div>
        </div>

        {/* Right Panel: Selected Hotspot Inspector + Recent Detections List */}
        <aside className="lg:col-span-4 xl:col-span-3 flex flex-col gap-4">
          {/* Selected Detection Inspector Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5 mb-3">
              <MapPin className="w-4 h-4 text-orange-400" />
              Selected Hotspot Telemetry
            </h2>

            {selectedFire ? (
              <div className="space-y-2.5 text-xs">
                <div className="flex items-center justify-between bg-slate-950/70 p-2.5 rounded-xl border border-slate-800">
                  <div>
                    <div className="text-[11px] text-slate-400">Coordinates (Lat, Lon)</div>
                    <div className="font-mono text-sm font-semibold text-white">
                      {selectedFire.latitude.toFixed(4)}°, {selectedFire.longitude.toFixed(4)}°
                    </div>
                  </div>
                  <span className="px-2 py-1 rounded-md text-[10px] font-bold uppercase bg-orange-500/20 text-orange-300 border border-orange-500/30">
                    {selectedFire.confidence || 'nominal'}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                    <div className="text-[10px] text-slate-400 flex items-center gap-1">
                      <Zap className="w-3 h-3 text-amber-400" /> Fire Radiative Power
                    </div>
                    <div className="text-sm font-bold text-amber-300 mt-0.5">
                      {selectedFire.frp !== null ? `${selectedFire.frp} MW` : 'N/A'}
                    </div>
                  </div>

                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                    <div className="text-[10px] text-slate-400 flex items-center gap-1">
                      <Thermometer className="w-3 h-3 text-rose-400" /> Brightness Temp
                    </div>
                    <div className="text-sm font-bold text-rose-300 mt-0.5">
                      {selectedFire.brightness !== null ? `${selectedFire.brightness} K` : 'N/A'}
                    </div>
                  </div>
                </div>

                <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80 space-y-1 text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Satellite / Sensor:</span>
                    <span className="text-slate-200 font-medium">{selectedFire.satellite}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Observation Time:</span>
                    <span className="text-slate-200 font-mono">
                      {selectedFire.acq_date} {selectedFire.acq_time.slice(0, 2)}:{selectedFire.acq_time.slice(2, 4)} UTC
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Scan × Track Pixel:</span>
                    <span className="text-slate-200">
                      {selectedFire.scan && selectedFire.track
                        ? `${selectedFire.scan} km × ${selectedFire.track} km`
                        : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500 py-4 text-center">
                Click any hotspot marker on the map or in the list below to inspect its telemetry.
              </p>
            )}
          </div>

          {/* Recent Detections Feed */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex-1 flex flex-col min-h-[280px] max-h-[420px]">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Flame className="w-4 h-4 text-orange-400" />
                Recent Detections
              </h2>
              <span className="text-[11px] text-slate-500">
                Top {recentDetections.length} shown
              </span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 pr-1">
              {recentDetections.map((fire) => {
                const isSelected = selectedFire?.id === fire.id;
                return (
                  <button
                    key={fire.id}
                    onClick={() => handleSelectDetection(fire)}
                    className={`w-full text-left p-2.5 rounded-xl border transition flex items-center justify-between gap-2 ${
                      isSelected
                        ? 'bg-orange-500/15 border-orange-500/50 text-white'
                        : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700 text-slate-200'
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="font-mono text-xs font-semibold truncate">
                        {fire.latitude.toFixed(3)}°, {fire.longitude.toFixed(3)}°
                      </div>
                      <div className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
                        <span>{fire.acq_date} {fire.acq_time.slice(0, 2)}:{fire.acq_time.slice(2, 4)} UTC</span>
                        <span>•</span>
                        <span className="truncate">{fire.satellite}</span>
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-xs font-bold text-amber-400">
                        {fire.frp !== null ? `${fire.frp} MW` : '—'}
                      </div>
                      <div className="text-[10px] uppercase text-slate-400">
                        {fire.confidence || 'nom'}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
