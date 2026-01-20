// =============================================================================
// Model Page - ML Model Documentation & Metrics
// =============================================================================
'use client';

import { ModelCard } from '@/components/layout/ModelCard';
import { Brain, Info, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ModelPage() {
  return (
    <div className="min-h-screen bg-[var(--void-black)] text-[var(--signal-white)]">
      <div className="scanline-overlay" />

      <div className="max-w-[1920px] mx-auto p-6">
        {/* Page Header */}
        <div className="flex items-center gap-4 mb-6">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[var(--accent-purple)] to-[var(--accent-cyan)] flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-wide">
              MODEL <span className="text-[var(--accent-cyan)]">DOCUMENTATION</span>
            </h1>
            <p className="text-xs text-[var(--signal-dim)] uppercase tracking-wider">
              LSTM Neural Network Specifications
            </p>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Full Model Card - 2 columns */}
          <div className="lg:col-span-2">
            <ModelCard variant="full" />
          </div>

          {/* Sidebar - Model Status & Info */}
          <div className="space-y-6">
            {/* Model Status */}
            <div className="glass-panel p-6">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle2 className="w-5 h-5 text-[var(--nominal-green)]" />
                <h3 className="text-sm font-semibold uppercase tracking-wider">
                  Model Status
                </h3>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-[var(--nominal-green)]/10 border border-[var(--nominal-green)]/30 rounded-lg">
                  <span className="text-xs">Runtime</span>
                  <span className="text-xs font-mono text-[var(--nominal-green)]">ONNX WebAssembly</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Version</span>
                  <span className="text-xs font-mono">v1.0.0</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Last Updated</span>
                  <span className="text-xs font-mono">2025-01-15</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-black/30 rounded-lg">
                  <span className="text-xs text-[var(--signal-dim)]">Checksum</span>
                  <span className="text-xs font-mono text-[var(--signal-grey)]">a7f3...</span>
                </div>
              </div>
            </div>

            {/* Important Notes */}
            <div className="glass-panel p-6">
              <div className="flex items-center gap-2 mb-4">
                <Info className="w-5 h-5 text-[var(--accent-cyan)]" />
                <h3 className="text-sm font-semibold uppercase tracking-wider">
                  Usage Notes
                </h3>
              </div>

              <div className="space-y-3 text-xs text-[var(--signal-dim)]">
                <p>
                  This model is optimized for real-time inference in browser environments
                  using WebAssembly acceleration.
                </p>
                <p>
                  Input telemetry must be normalized to [0,1] range for accurate predictions.
                  The preprocessing pipeline handles this automatically.
                </p>
                <p>
                  For production deployments, consider implementing model versioning
                  and A/B testing capabilities.
                </p>
              </div>
            </div>

            {/* Limitations Warning */}
            <div className="glass-panel p-6 border-l-4 border-[var(--warning-amber)]">
              <div className="flex items-center gap-2 mb-4">
                <AlertCircle className="w-5 h-5 text-[var(--warning-amber)]" />
                <h3 className="text-sm font-semibold uppercase tracking-wider text-[var(--warning-amber)]">
                  Known Limitations
                </h3>
              </div>

              <ul className="space-y-2 text-xs text-[var(--signal-dim)]">
                <li className="flex items-start gap-2">
                  <span className="text-[var(--warning-amber)]">•</span>
                  May produce false positives during orbital maneuvers
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[var(--warning-amber)]">•</span>
                  Reduced accuracy for novel failure modes
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[var(--warning-amber)]">•</span>
                  Requires 50-sample window for inference
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[var(--warning-amber)]">•</span>
                  Not validated for GEO orbit telemetry
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
