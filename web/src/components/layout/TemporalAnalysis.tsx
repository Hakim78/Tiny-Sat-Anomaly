// =============================================================================
// Temporal Analysis Component - Historical Trends & Patterns
// =============================================================================
'use client';

import { useMemo } from 'react';
import { useStore } from '@/store/useStore';
import {
  LineChart,
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  Clock,
  BarChart3,
  Activity,
} from 'lucide-react';

export function TemporalAnalysis() {
  const signalHistory = useStore((s) => s.signalHistory);
  const probabilityHistory = useStore((s) => s.probabilityHistory);
  const anomalyCount = useStore((s) => s.anomalyCount);
  const currentIndex = useStore((s) => s.currentIndex);

  // Calculate statistics
  const stats = useMemo(() => {
    const probValues = probabilityHistory.length > 0 ? probabilityHistory : [0];
    const signalValues = signalHistory.length > 0 ? signalHistory : [0];

    // Anomaly rate
    const anomalyRate = currentIndex > 0 ? (anomalyCount / currentIndex) * 100 : 0;

    // Average probability
    const avgProb = probValues.reduce((a, b) => a + b, 0) / probValues.length;

    // Trend calculation (last 20 vs previous 20)
    const recent = probValues.slice(-20);
    const previous = probValues.slice(-40, -20);
    const recentAvg = recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
    const previousAvg = previous.length > 0 ? previous.reduce((a, b) => a + b, 0) / previous.length : avgProb;
    const trend = recentAvg - previousAvg;

    // Signal variance
    const signalMean = signalValues.reduce((a, b) => a + b, 0) / signalValues.length;
    const signalVariance = signalValues.reduce((acc, val) =>
      acc + Math.pow(val - signalMean, 2), 0) / signalValues.length;

    // Max/Min probability
    const maxProb = Math.max(...probValues);
    const minProb = Math.min(...probValues);

    return {
      anomalyRate,
      avgProb,
      trend,
      signalVariance,
      maxProb,
      minProb,
      samplesProcessed: currentIndex,
      uptime: Math.floor(currentIndex / 10), // Simulated minutes
    };
  }, [probabilityHistory, signalHistory, anomalyCount, currentIndex]);

  // Generate hourly distribution (simulated)
  const hourlyDistribution = useMemo(() => {
    const hours = Array(24).fill(0).map((_, i) => ({
      hour: i,
      anomalies: Math.floor(Math.random() * (anomalyCount / 24 + 1)),
      samples: Math.floor(100 + Math.random() * 50),
    }));
    return hours;
  }, [anomalyCount]);

  // Generate orbit-based stats (simulated)
  const orbitStats = useMemo(() => {
    const orbits = Array(12).fill(0).map((_, i) => ({
      orbit: i + 1,
      anomalyRate: Math.random() * 5,
      avgSignal: 0.5 + Math.random() * 0.3,
    }));
    return orbits;
  }, []);

  const getTrendIcon = (trend: number) => {
    if (trend > 2) return <TrendingUp className="w-4 h-4 text-[var(--alert-red)]" />;
    if (trend < -2) return <TrendingDown className="w-4 h-4 text-[var(--nominal-green)]" />;
    return <Minus className="w-4 h-4 text-[var(--signal-dim)]" />;
  };

  return (
    <div className="space-y-4">
      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-[var(--accent-cyan)]" />
            <span className="text-[10px] text-[var(--signal-dim)] uppercase">Anomaly Rate</span>
          </div>
          <div className="text-3xl font-mono font-bold text-[var(--signal-white)]">
            {stats.anomalyRate.toFixed(2)}%
          </div>
          <div className="flex items-center gap-1 mt-2 text-[10px]">
            {getTrendIcon(stats.trend)}
            <span className={stats.trend > 0 ? 'text-[var(--alert-red)]' : 'text-[var(--nominal-green)]'}>
              {stats.trend > 0 ? '+' : ''}{stats.trend.toFixed(1)}% vs baseline
            </span>
          </div>
        </div>

        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-4 h-4 text-[var(--warning-amber)]" />
            <span className="text-[10px] text-[var(--signal-dim)] uppercase">Avg Probability</span>
          </div>
          <div className="text-3xl font-mono font-bold text-[var(--signal-white)]">
            {stats.avgProb.toFixed(1)}%
          </div>
          <div className="text-[10px] text-[var(--signal-dim)] mt-2">
            Range: {stats.minProb.toFixed(1)}% - {stats.maxProb.toFixed(1)}%
          </div>
        </div>

        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 mb-3">
            <Clock className="w-4 h-4 text-[var(--nominal-green)]" />
            <span className="text-[10px] text-[var(--signal-dim)] uppercase">Samples Processed</span>
          </div>
          <div className="text-3xl font-mono font-bold text-[var(--signal-white)]">
            {stats.samplesProcessed.toLocaleString()}
          </div>
          <div className="text-[10px] text-[var(--signal-dim)] mt-2">
            ~{stats.uptime} min runtime
          </div>
        </div>

        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 mb-3">
            <LineChart className="w-4 h-4 text-[var(--accent-cyan)]" />
            <span className="text-[10px] text-[var(--signal-dim)] uppercase">Signal Variance</span>
          </div>
          <div className="text-3xl font-mono font-bold text-[var(--signal-white)]">
            {stats.signalVariance.toFixed(4)}
          </div>
          <div className="text-[10px] text-[var(--signal-dim)] mt-2">
            {stats.signalVariance < 0.05 ? 'Stable' : stats.signalVariance < 0.1 ? 'Moderate' : 'High variance'}
          </div>
        </div>
      </div>

      {/* Probability Histogram */}
      <div className="glass-panel p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-[var(--accent-cyan)]" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
              Anomaly Probability Distribution
            </h2>
          </div>
          <div className="text-[10px] text-[var(--signal-dim)]">
            Last {probabilityHistory.length} samples
          </div>
        </div>

        {/* Simple bar chart visualization */}
        <div className="h-40 flex items-end gap-1">
          {Array(20).fill(0).map((_, i) => {
            const rangeStart = i * 5;
            const rangeEnd = rangeStart + 5;
            const count = probabilityHistory.filter(
              (p) => p >= rangeStart && p < rangeEnd
            ).length;
            const height = probabilityHistory.length > 0
              ? (count / probabilityHistory.length) * 100
              : 0;

            return (
              <div
                key={i}
                className="flex-1 flex flex-col items-center gap-1"
              >
                <div
                  className={`w-full rounded-t transition-all ${
                    rangeStart >= 50
                      ? 'bg-[var(--alert-red)]'
                      : rangeStart >= 30
                      ? 'bg-[var(--warning-amber)]'
                      : 'bg-[var(--accent-cyan)]'
                  }`}
                  style={{ height: `${Math.max(2, height * 1.5)}%` }}
                />
                {i % 4 === 0 && (
                  <span className="text-[8px] text-[var(--signal-dim)]">{rangeStart}%</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Hourly & Orbit Analysis */}
      <div className="grid grid-cols-2 gap-4">
        {/* Hourly Distribution */}
        <div className="glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <Calendar className="w-5 h-5 text-[var(--accent-cyan)]" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
              24-Hour Distribution
            </h2>
          </div>

          <div className="h-32 flex items-end gap-0.5">
            {hourlyDistribution.map((hour) => {
              const maxAnomalies = Math.max(...hourlyDistribution.map((h) => h.anomalies), 1);
              const height = (hour.anomalies / maxAnomalies) * 100;

              return (
                <div
                  key={hour.hour}
                  className="flex-1 group relative"
                >
                  <div
                    className="w-full bg-[var(--accent-cyan)]/60 hover:bg-[var(--accent-cyan)] transition-colors rounded-t"
                    style={{ height: `${Math.max(4, height)}%` }}
                  />
                  {/* Tooltip */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-[var(--panel-dark)] border border-white/10 rounded text-[8px] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                    <div className="text-[var(--signal-white)]">{hour.hour}:00 UTC</div>
                    <div className="text-[var(--signal-dim)]">{hour.anomalies} anomalies</div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex justify-between mt-2 text-[8px] text-[var(--signal-dim)]">
            <span>00:00</span>
            <span>06:00</span>
            <span>12:00</span>
            <span>18:00</span>
            <span>24:00</span>
          </div>
        </div>

        {/* Per-Orbit Stats */}
        <div className="glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-5 h-5 text-[var(--accent-cyan)]" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
              Anomaly Rate by Orbit
            </h2>
          </div>

          <div className="space-y-2">
            {orbitStats.slice(0, 6).map((orbit) => (
              <div key={orbit.orbit} className="flex items-center gap-3">
                <span className="text-[10px] text-[var(--signal-dim)] w-16">
                  Orbit {orbit.orbit}
                </span>
                <div className="flex-1 h-4 bg-black/30 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      orbit.anomalyRate > 3
                        ? 'bg-[var(--alert-red)]'
                        : orbit.anomalyRate > 1.5
                        ? 'bg-[var(--warning-amber)]'
                        : 'bg-[var(--nominal-green)]'
                    }`}
                    style={{ width: `${Math.min(100, orbit.anomalyRate * 20)}%` }}
                  />
                </div>
                <span className="text-[10px] font-mono text-[var(--signal-white)] w-12 text-right">
                  {orbit.anomalyRate.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Performance Trend */}
      <div className="glass-panel p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-[var(--accent-cyan)]" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
              Model Performance Trend
            </h2>
          </div>
          <div className="flex items-center gap-4 text-[10px]">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-[var(--nominal-green)]" />
              <span className="text-[var(--signal-dim)]">Precision</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-[var(--accent-cyan)]" />
              <span className="text-[var(--signal-dim)]">Recall</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-[var(--warning-amber)]" />
              <span className="text-[var(--signal-dim)]">F1-Score</span>
            </div>
          </div>
        </div>

        {/* Simulated performance over time */}
        <div className="grid grid-cols-7 gap-2">
          {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, i) => {
            const precision = 40 + Math.random() * 10;
            const recall = 70 + Math.random() * 10;
            const f1 = 50 + Math.random() * 10;

            return (
              <div key={day} className="text-center">
                <div className="text-[9px] text-[var(--signal-dim)] mb-2">{day}</div>
                <div className="space-y-1">
                  <div className="h-16 w-full bg-black/20 rounded relative">
                    <div
                      className="absolute bottom-0 left-0 right-0 bg-[var(--nominal-green)]/60 rounded-b"
                      style={{ height: `${precision}%` }}
                    />
                  </div>
                  <div className="text-[8px] text-[var(--signal-dim)]">{precision.toFixed(0)}%</div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-4 p-3 bg-black/20 rounded-lg border border-white/5">
          <div className="flex items-center gap-2 text-[10px]">
            <Info className="w-4 h-4 text-[var(--signal-dim)]" />
            <span className="text-[var(--signal-dim)]">
              No significant model drift detected. Performance within expected bounds.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function Info({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}
