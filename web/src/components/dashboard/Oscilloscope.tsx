// =============================================================================
// Oscilloscope - Elite Scientific Instrument Display
// Glassmorphism HUD with Medical/Aerospace Grade Visualization
// =============================================================================
'use client';

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import { useStore } from '@/store/useStore';
import { Radio, AlertTriangle, Activity, Waves } from 'lucide-react';

interface OscilloscopeProps {
  className?: string;
}

export function Oscilloscope({ className = '' }: OscilloscopeProps) {
  const signalHistory = useStore((s) => s.signalHistory);
  const anomalyIndices = useStore((s) => s.anomalyIndices);
  const isAnomaly = useStore((s) => s.isAnomaly);
  const currentIndex = useStore((s) => s.currentIndex);
  const anomalyProbability = useStore((s) => s.anomalyProbability);

  // Prepare chart data
  const chartData = useMemo(() => {
    return signalHistory.map((value, index) => ({
      index,
      value,
      isAnomaly: anomalyIndices.includes(index),
    }));
  }, [signalHistory, anomalyIndices]);

  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    if (signalHistory.length === 0) return [0, 1];
    const min = Math.min(...signalHistory);
    const max = Math.max(...signalHistory);
    const padding = (max - min) * 0.1 || 0.1;
    return [min - padding, max + padding];
  }, [signalHistory]);

  // Find anomaly ranges for highlighting
  const anomalyRanges = useMemo(() => {
    const ranges: { start: number; end: number }[] = [];
    let start: number | null = null;

    anomalyIndices.forEach((idx, i) => {
      if (start === null) {
        start = idx;
      }
      if (anomalyIndices[i + 1] !== idx + 1) {
        ranges.push({ start, end: idx });
        start = null;
      }
    });

    return ranges;
  }, [anomalyIndices]);

  // Signal stats
  const signalStats = useMemo(() => {
    if (signalHistory.length === 0) return { min: 0, max: 0, avg: 0, current: 0 };
    const min = Math.min(...signalHistory);
    const max = Math.max(...signalHistory);
    const avg = signalHistory.reduce((a, b) => a + b, 0) / signalHistory.length;
    const current = signalHistory[signalHistory.length - 1] || 0;
    return { min, max, avg, current };
  }, [signalHistory]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload[0]) return null;

    const data = payload[0].payload;
    return (
      <div className="glass-panel px-4 py-3 min-w-[140px]">
        <div className="text-[10px] text-[var(--signal-dim)] uppercase tracking-wider">
          Sample #{label}
        </div>
        <div className="text-xl font-mono font-semibold text-[var(--accent-cyan)] mt-1 text-glow-cyan">
          {data.value.toFixed(4)}
        </div>
        {data.isAnomaly && (
          <div className="flex items-center gap-1.5 text-[10px] text-[var(--alert-red)] font-bold mt-2 text-glow-red">
            <AlertTriangle className="w-3 h-3" />
            ANOMALY DETECTED
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={`oscilloscope h-full flex flex-col ${className}`}>
      {/* Header Bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isAnomaly ? 'bg-[var(--alert-red)] animate-pulse' : 'bg-[var(--nominal-green)]'}`} />
            <Waves className="w-4 h-4 text-[var(--accent-cyan)]" />
          </div>
          <div>
            <div className="text-xs font-semibold text-[var(--signal-white)]">
              TELEMETRY OSCILLOSCOPE
            </div>
            <div className="text-[9px] text-[var(--signal-dim)] uppercase tracking-wider">
              Channel 0 - Primary Signal
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6">
          {/* Live indicator */}
          <div className="flex items-center gap-2">
            <Radio className="w-3 h-3 text-[var(--nominal-green)] animate-pulse" />
            <span className="text-[10px] text-[var(--nominal-green)] uppercase font-semibold">
              Live
            </span>
          </div>

          {/* Anomaly indicator */}
          {isAnomaly && (
            <div className="flex items-center gap-2 px-3 py-1 rounded bg-[var(--alert-red)]/10 border border-[var(--alert-red)]/30">
              <AlertTriangle className="w-3 h-3 text-[var(--alert-red)] animate-pulse" />
              <span className="text-[10px] text-[var(--alert-red)] uppercase font-bold">
                Alert
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Stats Row */}
      <div className="flex items-center gap-6 px-4 py-2 border-b border-white/5 bg-black/20">
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-[var(--signal-dim)] uppercase">PKT</span>
          <span className="text-xs font-mono text-[var(--accent-cyan)]">
            {currentIndex.toString().padStart(6, '0')}
          </span>
        </div>
        <div className="w-px h-4 bg-white/10" />
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-[var(--signal-dim)] uppercase">MIN</span>
          <span className="text-xs font-mono text-[var(--signal-grey)]">
            {signalStats.min.toFixed(3)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-[var(--signal-dim)] uppercase">MAX</span>
          <span className="text-xs font-mono text-[var(--signal-grey)]">
            {signalStats.max.toFixed(3)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-[var(--signal-dim)] uppercase">AVG</span>
          <span className="text-xs font-mono text-[var(--signal-grey)]">
            {signalStats.avg.toFixed(3)}
          </span>
        </div>
        <div className="w-px h-4 bg-white/10" />
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-[var(--signal-dim)] uppercase">PROB</span>
          <span className={`text-xs font-mono font-semibold ${
            anomalyProbability > 0.5 ? 'text-[var(--alert-red)]' : 'text-[var(--nominal-green)]'
          }`}>
            {(anomalyProbability * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Chart Area */}
      <div className="flex-1 p-4 relative">
        {/* Grid overlay effect */}
        <div
          className="absolute inset-4 pointer-events-none opacity-30"
          style={{
            backgroundImage: `
              linear-gradient(rgba(0, 212, 255, 0.05) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0, 212, 255, 0.05) 1px, transparent 1px)
            `,
            backgroundSize: '20px 20px'
          }}
        />

        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
            {/* Main grid */}
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(0, 212, 255, 0.08)"
              vertical={true}
            />

            {/* Anomaly highlight areas */}
            {anomalyRanges.map((range, i) => (
              <ReferenceArea
                key={i}
                x1={range.start}
                x2={range.end}
                fill="var(--alert-red)"
                fillOpacity={0.15}
                stroke="var(--alert-red)"
                strokeOpacity={0.3}
              />
            ))}

            {/* Axes */}
            <XAxis
              dataKey="index"
              stroke="rgba(255, 255, 255, 0.1)"
              tick={{ fill: 'var(--signal-dim)', fontSize: 9, fontFamily: 'JetBrains Mono' }}
              tickLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
              axisLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={yDomain}
              stroke="rgba(255, 255, 255, 0.1)"
              tick={{ fill: 'var(--signal-dim)', fontSize: 9, fontFamily: 'JetBrains Mono' }}
              tickLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
              axisLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
              tickFormatter={(v) => v.toFixed(2)}
              width={50}
            />

            {/* Tooltip */}
            <Tooltip content={<CustomTooltip />} />

            {/* Center reference line */}
            <ReferenceLine
              y={0.5}
              stroke="rgba(0, 212, 255, 0.2)"
              strokeDasharray="8 4"
              label={{
                value: 'BASELINE',
                position: 'right',
                fill: 'var(--signal-dim)',
                fontSize: 8,
              }}
            />

            {/* Signal trace with glow effect */}
            <Line
              type="monotone"
              dataKey="value"
              stroke={isAnomaly ? 'var(--alert-red)' : 'var(--accent-cyan)'}
              strokeWidth={2}
              dot={false}
              activeDot={{
                r: 6,
                fill: isAnomaly ? 'var(--alert-red)' : 'var(--accent-cyan)',
                stroke: 'var(--void-black)',
                strokeWidth: 2,
              }}
              style={{
                filter: isAnomaly
                  ? 'drop-shadow(0 0 8px rgba(255, 59, 48, 0.8))'
                  : 'drop-shadow(0 0 6px rgba(0, 212, 255, 0.6))'
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-4 py-2 border-t border-white/5 bg-black/20">
        <div className="flex items-center gap-4 text-[9px]">
          <div className="flex items-center gap-1.5">
            <Activity className="w-3 h-3 text-[var(--accent-cyan)]" />
            <span className="text-[var(--signal-dim)] uppercase">Trace Active</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 bg-[var(--alert-red)]/40 rounded-sm" />
            <span className="text-[var(--signal-dim)] uppercase">Anomaly Zone</span>
          </div>
        </div>
        <div className="flex items-center gap-4 text-[9px]">
          <span className="text-[var(--signal-dim)]">
            BUFFER: <span className="text-[var(--signal-grey)] font-mono">{signalHistory.length}/150</span>
          </span>
          <span className="text-[var(--signal-dim)]">
            RATE: <span className="text-[var(--signal-grey)] font-mono">10 Hz</span>
          </span>
        </div>
      </div>
    </div>
  );
}

export default Oscilloscope;
