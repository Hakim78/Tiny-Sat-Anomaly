// =============================================================================
// Anomaly Breakdown Component - Shows contributing factors to anomaly score
// =============================================================================
'use client';

import { useMemo } from 'react';
import { useStore } from '@/store/useStore';
import { AlertTriangle, TrendingUp, Thermometer, Zap, Radio, HelpCircle } from 'lucide-react';

interface AnomalyFactor {
  id: string;
  name: string;
  contribution: number;
  icon: React.ReactNode;
  status: 'normal' | 'elevated' | 'critical';
  description: string;
}

export function AnomalyBreakdown() {
  const anomalyProbability = useStore((s) => s.anomalyProbability);
  const signalHistory = useStore((s) => s.signalHistory);
  const isSabotageActive = useStore((s) => s.isSabotageActive);

  // Compute simulated breakdown factors based on probability and signal variance
  const factors = useMemo<AnomalyFactor[]>(() => {
    const prob = anomalyProbability * 100;

    // Calculate signal variance for more realistic breakdown
    const recentSignals = signalHistory.slice(-20);
    const mean = recentSignals.length > 0
      ? recentSignals.reduce((a, b) => a + b, 0) / recentSignals.length
      : 0;
    const variance = recentSignals.length > 0
      ? recentSignals.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / recentSignals.length
      : 0;

    // Distribute the anomaly score across factors
    const basePower = isSabotageActive ? prob * 0.35 : prob * 0.25;
    const baseThermal = isSabotageActive ? prob * 0.25 : prob * 0.20;
    const baseAttitude = prob * 0.18 + variance * 10;
    const baseComm = prob * 0.12;
    const baseResidual = Math.max(0, prob - basePower - baseThermal - baseAttitude - baseComm);

    const getStatus = (val: number): 'normal' | 'elevated' | 'critical' => {
      if (val > 15) return 'critical';
      if (val > 8) return 'elevated';
      return 'normal';
    };

    return [
      {
        id: 'power',
        name: 'Power Fluctuation',
        contribution: Math.min(100, basePower + Math.random() * 3),
        icon: <Zap className="w-4 h-4" />,
        status: getStatus(basePower),
        description: 'Solar panel voltage variance',
      },
      {
        id: 'thermal',
        name: 'Thermal Drift',
        contribution: Math.min(100, baseThermal + Math.random() * 2),
        icon: <Thermometer className="w-4 h-4" />,
        status: getStatus(baseThermal),
        description: 'Temperature deviation from baseline',
      },
      {
        id: 'attitude',
        name: 'Attitude Jitter',
        contribution: Math.min(100, baseAttitude + Math.random() * 2),
        icon: <TrendingUp className="w-4 h-4" />,
        status: getStatus(baseAttitude),
        description: 'Orientation stability metric',
      },
      {
        id: 'comm',
        name: 'Comm Signal Quality',
        contribution: Math.min(100, baseComm + Math.random() * 1.5),
        icon: <Radio className="w-4 h-4" />,
        status: getStatus(baseComm),
        description: 'Downlink signal-to-noise ratio',
      },
      {
        id: 'residual',
        name: 'Residual Unexplained',
        contribution: Math.min(100, baseResidual + Math.random() * 1),
        icon: <HelpCircle className="w-4 h-4" />,
        status: getStatus(baseResidual),
        description: 'Unclassified anomaly factors',
      },
    ];
  }, [anomalyProbability, signalHistory, isSabotageActive]);

  const totalContribution = factors.reduce((sum, f) => sum + f.contribution, 0);
  const confidence = anomalyProbability > 0.5 ? 'HIGH' : anomalyProbability > 0.2 ? 'MEDIUM' : 'LOW';

  return (
    <div className="glass-panel p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <AlertTriangle className={`w-4 h-4 ${anomalyProbability > 0.3 ? 'text-[var(--alert-red)]' : 'text-[var(--warning-amber)]'}`} />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--signal-white)]">
            Anomaly Breakdown
          </h3>
        </div>
        <div className={`px-2 py-0.5 rounded text-[8px] font-bold ${
          confidence === 'HIGH'
            ? 'bg-[var(--alert-red)]/20 text-[var(--alert-red)]'
            : confidence === 'MEDIUM'
            ? 'bg-[var(--warning-amber)]/20 text-[var(--warning-amber)]'
            : 'bg-[var(--nominal-green)]/20 text-[var(--nominal-green)]'
        }`}>
          CONFIDENCE: {confidence}
        </div>
      </div>

      <div className="space-y-3">
        {factors.map((factor) => (
          <div key={factor.id} className="group">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className={`${
                  factor.status === 'critical'
                    ? 'text-[var(--alert-red)]'
                    : factor.status === 'elevated'
                    ? 'text-[var(--warning-amber)]'
                    : 'text-[var(--signal-dim)]'
                }`}>
                  {factor.icon}
                </span>
                <span className="text-[10px] text-[var(--signal-white)]">{factor.name}</span>
              </div>
              <span className={`text-[10px] font-mono font-bold ${
                factor.status === 'critical'
                  ? 'text-[var(--alert-red)]'
                  : factor.status === 'elevated'
                  ? 'text-[var(--warning-amber)]'
                  : 'text-[var(--signal-dim)]'
              }`}>
                +{factor.contribution.toFixed(1)}%
              </span>
            </div>

            {/* Progress bar */}
            <div className="h-1.5 bg-black/30 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  factor.status === 'critical'
                    ? 'bg-[var(--alert-red)]'
                    : factor.status === 'elevated'
                    ? 'bg-[var(--warning-amber)]'
                    : 'bg-[var(--accent-cyan)]/50'
                }`}
                style={{ width: `${Math.min(100, factor.contribution * 2)}%` }}
              />
            </div>

            {/* Description on hover */}
            <p className="text-[8px] text-[var(--signal-dim)] mt-0.5 opacity-60">
              {factor.description}
            </p>
          </div>
        ))}
      </div>

      {/* Total */}
      <div className="mt-4 pt-3 border-t border-white/10">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-[var(--signal-dim)] uppercase">Total Anomaly Index</span>
          <span className={`text-sm font-mono font-bold ${
            anomalyProbability > 0.5
              ? 'text-[var(--alert-red)]'
              : anomalyProbability > 0.3
              ? 'text-[var(--warning-amber)]'
              : 'text-[var(--nominal-green)]'
          }`}>
            {(anomalyProbability * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}
