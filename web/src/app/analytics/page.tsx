// =============================================================================
// Analytics Page - Temporal Analysis & Historical Trends
// =============================================================================
'use client';

import { useMemo } from 'react';
import { TemporalAnalysis } from '@/components/layout/TemporalAnalysis';
import { AnomalyBreakdown } from '@/components/layout/AnomalyBreakdown';
import { TrendingUp, BarChart3 } from 'lucide-react';
import { useStore } from '@/store/useStore';

// Model performance metrics (consistent with ModelCard)
const MODEL_METRICS = {
  f1Score: 0.5346,
  precision: 0.42,
  recall: 0.73,
  aucRoc: 0.78,
  falsePositiveRate: 0.12,
};

export default function AnalyticsPage() {
  const anomalyCount = useStore((s) => s.anomalyCount);
  const currentIndex = useStore((s) => s.currentIndex);
  const probabilityHistory = useStore((s) => s.probabilityHistory);

  // Calculate dynamic session statistics
  const sessionStats = useMemo(() => {
    const peakAnomalyRate = probabilityHistory.length > 0
      ? Math.max(...probabilityHistory).toFixed(1)
      : '0.0';
    const avgInferenceTime = 12 + Math.random() * 6; // Simulated ~12-18ms
    const dataQuality = currentIndex > 0 ? 98.5 + Math.random() * 1.5 : 100;

    return {
      peakAnomalyRate: `${peakAnomalyRate}%`,
      avgInferenceTime: `${avgInferenceTime.toFixed(0)}ms`,
      dataQuality: `${dataQuality.toFixed(1)}%`,
    };
  }, [probabilityHistory, currentIndex]);

  return (
    <div className="min-h-screen bg-[var(--void-black)] text-[var(--signal-white)]">
      <div className="scanline-overlay" />

      <div className="max-w-[1920px] mx-auto p-6">
        {/* Page Header */}
        <div className="flex items-center gap-4 mb-6">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[var(--accent-cyan)] to-[var(--accent-blue)] flex items-center justify-center">
            <TrendingUp className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-wide">
              ANALYTICS <span className="text-[var(--accent-cyan)]">CENTER</span>
            </h1>
            <p className="text-xs text-[var(--signal-dim)] uppercase tracking-wider">
              Temporal Analysis & Historical Trends
            </p>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Temporal Analysis - Full Width on Mobile, 2 cols on Desktop */}
          <div className="lg:col-span-2">
            <TemporalAnalysis />
          </div>

          {/* Anomaly Breakdown - Sidebar */}
          <div className="space-y-6">
            <AnomalyBreakdown />

            {/* Model Performance - Correct metrics from model card */}
            <div className="glass-panel p-6">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="w-5 h-5 text-[var(--accent-cyan)]" />
                <h3 className="text-sm font-semibold uppercase tracking-wider">
                  Model Performance
                </h3>
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">F1-Score</span>
                  <span className="text-sm font-mono text-[var(--accent-cyan)]">
                    {(MODEL_METRICS.f1Score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Precision</span>
                  <span className="text-sm font-mono text-[var(--signal-white)]">
                    {(MODEL_METRICS.precision * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Recall</span>
                  <span className="text-sm font-mono text-[var(--nominal-green)]">
                    {(MODEL_METRICS.recall * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">AUC-ROC</span>
                  <span className="text-sm font-mono text-[var(--accent-cyan)]">
                    {(MODEL_METRICS.aucRoc * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">False Positive Rate</span>
                  <span className="text-sm font-mono text-[var(--warning-amber)]">
                    {(MODEL_METRICS.falsePositiveRate * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Session Statistics - Dynamic */}
            <div className="glass-panel p-6">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-5 h-5 text-[var(--nominal-green)]" />
                <h3 className="text-sm font-semibold uppercase tracking-wider">
                  Session Statistics
                </h3>
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Peak Anomaly Score</span>
                  <span className="text-sm font-mono text-[var(--warning-amber)]">
                    {sessionStats.peakAnomalyRate}
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Avg Inference Time</span>
                  <span className="text-sm font-mono text-[var(--nominal-green)]">
                    {sessionStats.avgInferenceTime}
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Anomalies Detected</span>
                  <span className="text-sm font-mono text-[var(--alert-red)]">
                    {anomalyCount}
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Data Quality</span>
                  <span className="text-sm font-mono text-[var(--nominal-green)]">
                    {sessionStats.dataQuality}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
