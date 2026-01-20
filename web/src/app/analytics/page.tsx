// =============================================================================
// Analytics Page - Temporal Analysis & Historical Trends
// =============================================================================
'use client';

import { TemporalAnalysis } from '@/components/layout/TemporalAnalysis';
import { AnomalyBreakdown } from '@/components/layout/AnomalyBreakdown';
import { TrendingUp, BarChart3 } from 'lucide-react';

export default function AnalyticsPage() {
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

            {/* Quick Stats */}
            <div className="glass-panel p-6">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="w-5 h-5 text-[var(--accent-cyan)]" />
                <h3 className="text-sm font-semibold uppercase tracking-wider">
                  Session Statistics
                </h3>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Peak Anomaly Rate</span>
                  <span className="text-sm font-mono text-[var(--warning-amber)]">12.4%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Avg Response Time</span>
                  <span className="text-sm font-mono text-[var(--nominal-green)]">23ms</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Model Accuracy</span>
                  <span className="text-sm font-mono text-[var(--accent-cyan)]">94.2%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Data Quality</span>
                  <span className="text-sm font-mono text-[var(--nominal-green)]">98.7%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
