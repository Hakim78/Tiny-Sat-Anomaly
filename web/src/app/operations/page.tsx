// =============================================================================
// Operations Page - SOP & Operator Management
// =============================================================================
'use client';

import { SOPPanel } from '@/components/layout/SOPPanel';
import { AnomalyBreakdown } from '@/components/layout/AnomalyBreakdown';
import { Shield, Users, Clock, AlertTriangle } from 'lucide-react';
import { useStore } from '@/store/useStore';

export default function OperationsPage() {
  const anomalyCount = useStore((s) => s.anomalyCount);
  const isAnomaly = useStore((s) => s.isAnomaly);

  return (
    <div className="min-h-screen bg-[var(--void-black)] text-[var(--signal-white)]">
      <div className="scanline-overlay" />

      <div className="max-w-[1920px] mx-auto p-6">
        {/* Page Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[var(--warning-amber)] to-[var(--alert-red)] flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-wide">
                OPERATIONS <span className="text-[var(--accent-cyan)]">CENTER</span>
              </h1>
              <p className="text-xs text-[var(--signal-dim)] uppercase tracking-wider">
                Standard Operating Procedures & Response Management
              </p>
            </div>
          </div>

          {/* Alert Status */}
          {isAnomaly && (
            <div className="flex items-center gap-3 px-4 py-2 bg-[var(--alert-red)]/20 border border-[var(--alert-red)]/30 rounded-lg animate-pulse">
              <AlertTriangle className="w-5 h-5 text-[var(--alert-red)]" />
              <span className="text-sm font-semibold text-[var(--alert-red)]">
                ACTIVE ANOMALY - RESPONSE REQUIRED
              </span>
            </div>
          )}
        </div>

        {/* Quick Stats Bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="glass-panel p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-[var(--accent-cyan)]/10 flex items-center justify-center">
              <Users className="w-5 h-5 text-[var(--accent-cyan)]" />
            </div>
            <div>
              <div className="text-[10px] text-[var(--signal-dim)] uppercase">Operators Online</div>
              <div className="text-lg font-mono font-bold">3</div>
            </div>
          </div>

          <div className="glass-panel p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-[var(--warning-amber)]/10 flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-[var(--warning-amber)]" />
            </div>
            <div>
              <div className="text-[10px] text-[var(--signal-dim)] uppercase">Anomalies Today</div>
              <div className="text-lg font-mono font-bold text-[var(--warning-amber)]">{anomalyCount}</div>
            </div>
          </div>

          <div className="glass-panel p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-[var(--nominal-green)]/10 flex items-center justify-center">
              <Shield className="w-5 h-5 text-[var(--nominal-green)]" />
            </div>
            <div>
              <div className="text-[10px] text-[var(--signal-dim)] uppercase">SOPs Executed</div>
              <div className="text-lg font-mono font-bold text-[var(--nominal-green)]">7</div>
            </div>
          </div>

          <div className="glass-panel p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-[var(--accent-purple)]/10 flex items-center justify-center">
              <Clock className="w-5 h-5 text-[var(--accent-purple)]" />
            </div>
            <div>
              <div className="text-[10px] text-[var(--signal-dim)] uppercase">Avg Response</div>
              <div className="text-lg font-mono font-bold">4.2m</div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* SOP Panel - Full Width on Mobile, 2 cols on Desktop */}
          <div className="lg:col-span-2">
            <SOPPanel />
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Current Anomaly Breakdown */}
            <AnomalyBreakdown />

            {/* Shift Handover Notes */}
            <div className="glass-panel p-6">
              <h3 className="text-sm font-semibold uppercase tracking-wider mb-4 flex items-center gap-2">
                <Clock className="w-4 h-4 text-[var(--accent-cyan)]" />
                Shift Handover Notes
              </h3>

              <div className="space-y-3 text-xs">
                <div className="p-3 bg-black/30 rounded-lg">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[var(--signal-dim)]">From: MCO-L1-CHEN</span>
                    <span className="text-[var(--signal-dim)] font-mono">06:00 UTC</span>
                  </div>
                  <p className="text-[var(--signal-grey)]">
                    Nominal operations throughout night shift. Minor thermal excursion
                    at 03:42 UTC resolved via SOP-T09. All subsystems green.
                  </p>
                </div>

                <div className="p-3 bg-[var(--warning-amber)]/10 border border-[var(--warning-amber)]/30 rounded-lg">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[var(--warning-amber)]">Watch Item</span>
                    <span className="text-[var(--signal-dim)] font-mono">Active</span>
                  </div>
                  <p className="text-[var(--signal-grey)]">
                    Power subsystem showing intermittent fluctuations. Monitor closely
                    during eclipse transitions.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
