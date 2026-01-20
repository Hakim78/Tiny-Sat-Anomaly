// =============================================================================
// Main Page - TINY-SAT Mission Control Dashboard v2.0
// Elite Glassmorphism HUD - SpaceX Dragon / The Expanse Style
// =============================================================================
'use client';

import { useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useStore, TelemetryData } from '@/store/useStore';
import { useSimulationLoop } from '@/hooks/useInference';
import { useSatelliteTracking, getN2YOApiKey } from '@/hooks/useSatelliteTracking';
import { ControlPanel } from '@/components/layout/ControlPanel';
import { StatusBadge } from '@/components/layout/StatusBadge';
import { MetricCard } from '@/components/layout/MetricCard';
import { MissionLog } from '@/components/layout/MissionLog';
import { ModelMetrics } from '@/components/layout/ModelMetrics';
import { DataAttribution } from '@/components/layout/DataAttribution';
import { Oscilloscope } from '@/components/dashboard/Oscilloscope';
import {
  Satellite,
  Gauge,
  Navigation,
  Thermometer,
  Zap,
  Signal,
  Globe2,
} from 'lucide-react';

// Dynamic import for 3D Scene (client-side only)
const Scene3D = dynamic(
  () => import('@/components/three/Scene3D').then((mod) => mod.Scene3D),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-[var(--void-black)]">
        <div className="flex flex-col items-center gap-4">
          <div className="relative">
            <Globe2 className="w-16 h-16 text-[var(--accent-cyan)] animate-pulse" />
            <div className="absolute inset-0 w-16 h-16 border-2 border-[var(--accent-cyan)]/30 rounded-full animate-ping" />
          </div>
          <div className="text-center">
            <div className="text-xs text-[var(--signal-grey)] uppercase tracking-wider">
              Initializing 3D Scene
            </div>
            <div className="skeleton w-32 h-1 mt-2 mx-auto" />
          </div>
        </div>
      </div>
    ),
  }
);

export default function MissionControlPage() {
  const setTelemetryData = useStore((s) => s.setTelemetryData);
  const addLog = useStore((s) => s.addLog);
  const setLiveTracking = useStore((s) => s.setLiveTracking);
  const anomalyProbability = useStore((s) => s.anomalyProbability);
  const anomalyThreshold = useStore((s) => s.anomalyThreshold);
  const satellitePosition = useStore((s) => s.satellitePosition);
  const isAnomaly = useStore((s) => s.isAnomaly);
  const currentIndex = useStore((s) => s.currentIndex);
  const isPlaying = useStore((s) => s.isPlaying);
  const isLiveTracking = useStore((s) => s.isLiveTracking);

  // Initialize simulation loop
  useSimulationLoop();

  // Initialize satellite tracking (API or simulated)
  const { isLive, isConnected, satelliteName } = useSatelliteTracking({
    apiKey: getN2YOApiKey(),
    updateInterval: 5000, // Update every 5 seconds
    enabled: true,
  });

  // Sync live tracking state to store
  useEffect(() => {
    setLiveTracking(isLive);
  }, [isLive, setLiveTracking]);

  // Load telemetry data on mount
  useEffect(() => {
    async function loadData() {
      try {
        addLog('INFO', 'Initializing telemetry stream...');

        const response = await fetch('/telemetry_data.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data: TelemetryData = await response.json();
        setTelemetryData(data);

        addLog('SUCCESS', `Telemetry online: ${data.metadata.samples} packets`);
      } catch (error) {
        console.error('Failed to load telemetry data:', error);
        addLog('ERROR', 'Telemetry stream failed - fallback mode');

        const syntheticData: TelemetryData = {
          metadata: { source: 'Synthetic', samples: 2000, features: 25, normalized: true },
          data: Array.from({ length: 2000 }, (_, i) =>
            Array.from({ length: 25 }, (_, j) =>
              Math.sin(i * 0.02 * (j + 1)) * 0.5 + 0.5 + Math.random() * 0.1
            )
          ),
        };
        setTelemetryData(syntheticData);
      }
    }
    loadData();
  }, [setTelemetryData, addLog]);

  // Format mission elapsed time
  const formatMET = (index: number) => {
    const hours = Math.floor(index / 3600).toString().padStart(2, '0');
    const minutes = Math.floor((index % 3600) / 60).toString().padStart(2, '0');
    const seconds = (index % 60).toString().padStart(2, '0');
    return `T+${hours}:${minutes}:${seconds}`;
  };

  return (
    <div className="min-h-screen bg-[var(--void-black)] text-[var(--signal-white)] overflow-hidden">
      {/* Scanline overlay */}
      <div className="scanline-overlay" />

      {/* ===== MAIN CONTENT ===== */}
      <div className="pb-[40px] min-h-screen">
        <div className="h-full grid grid-cols-1 md:grid-cols-12 gap-3 p-3">

          {/* LEFT SIDEBAR - Controls & Metrics */}
          <aside className="order-2 md:order-1 col-span-1 md:col-span-3 lg:col-span-2 flex flex-col gap-3">
            <ControlPanel />

            {/* Quick Metrics */}
            <div className="grid grid-cols-4 md:grid-cols-2 gap-2">
              <MetricCard
                label="Latitude"
                value={satellitePosition.lat.toFixed(1)}
                suffix="°"
                icon={<Navigation className="w-4 h-4" />}
              />
              <MetricCard
                label="Longitude"
                value={satellitePosition.lon.toFixed(1)}
                suffix="°"
                icon={<Globe2 className="w-4 h-4" />}
              />
              <MetricCard
                label="Altitude"
                value={satellitePosition.alt.toFixed(0)}
                suffix=" km"
                icon={<Signal className="w-4 h-4" />}
              />
              <MetricCard
                label="Probability"
                value={(anomalyProbability * 100).toFixed(1)}
                suffix="%"
                icon={<Gauge className="w-4 h-4" />}
                variant={isAnomaly ? 'danger' : 'default'}
              />
            </div>
          </aside>

          {/* CENTER - 3D Scene & Oscilloscope */}
          <section className="order-1 md:order-2 col-span-1 md:col-span-9 lg:col-span-7 flex flex-col gap-3">
            {/* 3D Scene Hero */}
            <div className="h-[50vh] md:h-[55vh] lg:flex-1 lg:min-h-[400px] relative rounded-xl overflow-hidden glass-panel hud-frame">
              <Scene3D className="absolute inset-0" />

              {/* HUD Overlays */}
              <div className="absolute inset-0 pointer-events-none">
                {/* Top-left: Position */}
                <div className="absolute top-4 left-4 glass-panel px-4 py-3 pointer-events-auto">
                  <div className="flex items-center gap-2 text-[9px] text-[var(--signal-dim)] uppercase tracking-wider mb-1">
                    <Navigation className="w-3 h-3 text-[var(--accent-cyan)]" />
                    Orbital Position
                  </div>
                  <div className="font-mono text-sm">
                    <span className="text-[var(--accent-cyan)]">{satellitePosition.lat.toFixed(2)}°N</span>
                    <span className="text-[var(--signal-dim)] mx-2">/</span>
                    <span className="text-[var(--accent-cyan)]">{satellitePosition.lon.toFixed(2)}°E</span>
                  </div>
                  <div className="text-[9px] text-[var(--signal-dim)] mt-1 font-mono">
                    ALT: {satellitePosition.alt.toFixed(0)} km
                  </div>
                </div>

                {/* Top-right: Anomaly Gauge */}
                <div className="absolute top-4 right-4 glass-panel px-4 py-3 pointer-events-auto min-w-[160px]">
                  <div className="flex items-center gap-2 text-[9px] text-[var(--signal-dim)] uppercase tracking-wider mb-2">
                    <Gauge className="w-3 h-3 text-[var(--accent-cyan)]" />
                    Anomaly Index
                  </div>
                  <div className="progress-bar mb-2">
                    <div
                      className={`fill ${anomalyProbability > anomalyThreshold ? 'danger' : ''}`}
                      style={{ width: `${anomalyProbability * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between items-baseline">
                    <span className={`text-xl font-bold font-mono ${
                      anomalyProbability > anomalyThreshold ? 'text-[var(--alert-red)] text-glow-red' : 'text-[var(--accent-cyan)] text-glow-cyan'
                    }`}>
                      {(anomalyProbability * 100).toFixed(1)}%
                    </span>
                    <span className="text-[9px] text-[var(--signal-dim)]">
                      THR: {(anomalyThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Bottom-center: Mission Time */}
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 glass-panel px-6 py-2 pointer-events-auto">
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span className="text-[var(--signal-dim)]">MET</span>
                    <span className="text-[var(--accent-cyan)] tracking-wider text-glow-cyan">
                      {formatMET(currentIndex)}
                    </span>
                    <span className="text-[var(--signal-dim)]">|</span>
                    <span className="text-[var(--signal-grey)]">
                      PKT {currentIndex.toString().padStart(6, '0')}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Oscilloscope */}
            <div className="h-56 md:h-64 lg:h-72 rounded-xl overflow-hidden">
              <Oscilloscope />
            </div>
          </section>

          {/* RIGHT SIDEBAR - Status & Logs */}
          <aside className="order-3 col-span-1 md:col-span-12 lg:col-span-3 grid grid-cols-1 md:grid-cols-3 lg:grid-cols-1 gap-3">
            {/* Spacecraft Info Card */}
            <div className="glass-panel-accent p-4">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-[var(--accent-cyan)]/10 flex items-center justify-center">
                  <Satellite className={`w-5 h-5 ${isAnomaly ? 'text-[var(--alert-red)]' : 'text-[var(--accent-cyan)]'}`} />
                </div>
                <div className="flex-1">
                  <div className="text-sm font-semibold">{satelliteName} Spacecraft</div>
                  <div className="text-[9px] text-[var(--signal-dim)] uppercase tracking-wider">
                    {isLiveTracking ? 'Live Tracking Active' : 'Simulation Mode'}
                  </div>
                </div>
                {isConnected && (
                  <div className="w-2 h-2 rounded-full bg-[var(--nominal-green)] animate-pulse" />
                )}
              </div>

              <div className="grid grid-cols-4 md:grid-cols-2 gap-2">
                <div className="bg-black/30 rounded-lg p-2 md:p-3">
                  <div className="text-[8px] md:text-[9px] text-[var(--signal-dim)] uppercase mb-1">Orbit</div>
                  <div className="text-xs md:text-sm font-mono text-[var(--signal-white)]">LEO</div>
                </div>
                <div className="bg-black/30 rounded-lg p-2 md:p-3">
                  <div className="text-[8px] md:text-[9px] text-[var(--signal-dim)] uppercase mb-1">Period</div>
                  <div className="text-xs md:text-sm font-mono text-[var(--signal-white)]">98 min</div>
                </div>
                <div className="bg-black/30 rounded-lg p-2 md:p-3">
                  <div className="text-[8px] md:text-[9px] text-[var(--signal-dim)] uppercase mb-1">Inclination</div>
                  <div className="text-xs md:text-sm font-mono text-[var(--signal-white)]">98.1°</div>
                </div>
                <div className="bg-black/30 rounded-lg p-2 md:p-3">
                  <div className="text-[8px] md:text-[9px] text-[var(--signal-dim)] uppercase mb-1">Launch</div>
                  <div className="text-xs md:text-sm font-mono text-[var(--signal-white)]">2015</div>
                </div>
              </div>
            </div>

            {/* Status & Metrics Row */}
            <div className="flex flex-col gap-3">
              <StatusBadge />
              <div className="grid grid-cols-2 gap-2">
                <MetricCard
                  label="Temperature"
                  value="23.4"
                  suffix="°C"
                  icon={<Thermometer className="w-4 h-4" />}
                />
                <MetricCard
                  label="Power"
                  value="1.2"
                  suffix=" kW"
                  icon={<Zap className="w-4 h-4" />}
                  variant="success"
                />
              </div>
            </div>

            {/* ML Model Metrics - Important for professional presentation */}
            <div className="hidden lg:block">
              <ModelMetrics compact />
            </div>

            {/* Data Attribution - Credits & Sources */}
            <div className="hidden lg:block">
              <DataAttribution />
            </div>

            {/* Mission Log */}
            <div className="min-h-[150px] md:min-h-[180px]">
              <MissionLog maxEntries={8} />
            </div>
          </aside>
        </div>
      </div>

      {/* ===== FOOTER ===== */}
      <footer className="fixed bottom-0 left-0 right-0 z-40 border-t border-white/5 bg-[var(--void-black)]/90 backdrop-blur-sm">
        <div className="max-w-[1920px] mx-auto px-4 md:px-6 py-2 flex flex-col md:flex-row items-center justify-between text-[8px] md:text-[9px] text-[var(--signal-dim)] gap-1 md:gap-0">
          <div className="flex items-center gap-2 md:gap-4">
            <span>TINY-SAT v2.0</span>
            <span className="hidden md:inline text-[var(--signal-dim)]">|</span>
            <span className="hidden md:inline">LSTM Neural Network</span>
            <span className="hidden md:inline text-[var(--signal-dim)]">|</span>
            <span>ONNX Runtime</span>
          </div>
          <div className="flex items-center gap-2 md:gap-4 font-mono">
            <span className="hidden md:inline">NASA SMAP</span>
            <span className="text-[var(--accent-cyan)]">
              {new Date().toISOString().split('T')[0]}
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
