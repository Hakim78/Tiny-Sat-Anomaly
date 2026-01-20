// =============================================================================
// Model Card Component - ML Model Documentation & Transparency
// =============================================================================
'use client';

import {
  Brain,
  Database,
  AlertCircle,
  Clock,
  CheckCircle,
  XCircle,
  Info,
  GitBranch,
  Cpu,
  TrendingUp,
} from 'lucide-react';

interface ModelCardProps {
  compact?: boolean;
}

export function ModelCard({ compact = false }: ModelCardProps) {
  // Model metadata - would come from config in production
  const modelInfo = {
    name: 'LSTM-AD',
    version: '2.3.1',
    type: 'Long Short-Term Memory Autoencoder',
    framework: 'ONNX Runtime WebAssembly',
    inputShape: '[1, 50, 25]',
    outputShape: '[1, 1]',
    parameters: '~125K',
    latency: '<15ms',
    lastRetrain: '2025-12-08',
    trainingData: {
      source: 'NASA SMAP Telemetry Dataset',
      samples: '18 months historical data',
      features: 25,
      anomalyRatio: '4.2%',
    },
    performance: {
      f1Score: 0.5346,
      precision: 0.42,
      recall: 0.73,
      aucRoc: 0.78,
      falsePositiveRate: 0.12,
    },
    blindSpots: [
      'Eclipse phase transitions',
      'Safe-mode recovery sequences',
      'Antenna deployment events',
      'Solar flare direct impact (>X5)',
    ],
    validityConditions: [
      'LEO orbit (400-800km altitude)',
      'Normal operational mode',
      'Telemetry sampling rate >= 1Hz',
      'At least 50 consecutive samples',
    ],
  };

  if (compact) {
    return (
      <div className="glass-panel p-4">
        <div className="flex items-center gap-2 mb-3">
          <Brain className="w-4 h-4 text-[var(--accent-cyan)]" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--signal-white)]">
            ML Model
          </h3>
          <span className="ml-auto text-[9px] font-mono text-[var(--accent-cyan)]">
            v{modelInfo.version}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[9px]">
          <div className="bg-black/20 rounded p-2">
            <div className="text-[var(--signal-dim)]">Model</div>
            <div className="text-[var(--signal-white)] font-mono">{modelInfo.name}</div>
          </div>
          <div className="bg-black/20 rounded p-2">
            <div className="text-[var(--signal-dim)]">Latency</div>
            <div className="text-[var(--nominal-green)] font-mono">{modelInfo.latency}</div>
          </div>
          <div className="bg-black/20 rounded p-2">
            <div className="text-[var(--signal-dim)]">F1-Score</div>
            <div className="text-[var(--signal-white)] font-mono">{(modelInfo.performance.f1Score * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-black/20 rounded p-2">
            <div className="text-[var(--signal-dim)]">Recall</div>
            <div className="text-[var(--signal-white)] font-mono">{(modelInfo.performance.recall * 100).toFixed(0)}%</div>
          </div>
        </div>

        <div className="mt-3 pt-2 border-t border-white/5">
          <div className="flex items-center gap-1 text-[8px] text-[var(--signal-dim)]">
            <Clock className="w-3 h-3" />
            <span>Last retrain: {modelInfo.lastRetrain}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="glass-panel p-6">
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-xl bg-[var(--accent-cyan)]/10 border border-[var(--accent-cyan)]/30 flex items-center justify-center">
              <Brain className="w-7 h-7 text-[var(--accent-cyan)]" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-[var(--signal-white)]">
                {modelInfo.name}
                <span className="ml-2 text-sm font-mono text-[var(--accent-cyan)]">v{modelInfo.version}</span>
              </h2>
              <p className="text-sm text-[var(--signal-dim)]">{modelInfo.type}</p>
              <div className="flex items-center gap-2 mt-1">
                <span className="px-2 py-0.5 text-[9px] rounded bg-[var(--nominal-green)]/20 text-[var(--nominal-green)]">
                  PRODUCTION
                </span>
                <span className="px-2 py-0.5 text-[9px] rounded bg-[var(--accent-cyan)]/20 text-[var(--accent-cyan)]">
                  {modelInfo.framework}
                </span>
              </div>
            </div>
          </div>

          <div className="text-right">
            <div className="text-[10px] text-[var(--signal-dim)]">Last Retrain</div>
            <div className="text-sm font-mono text-[var(--signal-white)]">{modelInfo.lastRetrain}</div>
            <div className="text-[9px] text-[var(--warning-amber)] mt-1">
              T-42 days ago
            </div>
          </div>
        </div>

        {/* Technical Specs */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-black/20 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <Cpu className="w-4 h-4 text-[var(--accent-cyan)]" />
              <span className="text-[10px] text-[var(--signal-dim)] uppercase">Input Shape</span>
            </div>
            <div className="text-sm font-mono text-[var(--signal-white)]">{modelInfo.inputShape}</div>
          </div>
          <div className="bg-black/20 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-[var(--accent-cyan)]" />
              <span className="text-[10px] text-[var(--signal-dim)] uppercase">Output Shape</span>
            </div>
            <div className="text-sm font-mono text-[var(--signal-white)]">{modelInfo.outputShape}</div>
          </div>
          <div className="bg-black/20 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <GitBranch className="w-4 h-4 text-[var(--accent-cyan)]" />
              <span className="text-[10px] text-[var(--signal-dim)] uppercase">Parameters</span>
            </div>
            <div className="text-sm font-mono text-[var(--signal-white)]">{modelInfo.parameters}</div>
          </div>
          <div className="bg-black/20 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4 text-[var(--nominal-green)]" />
              <span className="text-[10px] text-[var(--signal-dim)] uppercase">Latency</span>
            </div>
            <div className="text-sm font-mono text-[var(--nominal-green)]">{modelInfo.latency}</div>
          </div>
        </div>
      </div>

      {/* Training Data */}
      <div className="glass-panel p-6">
        <div className="flex items-center gap-2 mb-4">
          <Database className="w-5 h-5 text-[var(--accent-cyan)]" />
          <h3 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
            Training Data
          </h3>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-[10px] text-[var(--signal-dim)] uppercase mb-1">Source</div>
            <div className="text-sm text-[var(--signal-white)]">{modelInfo.trainingData.source}</div>
          </div>
          <div>
            <div className="text-[10px] text-[var(--signal-dim)] uppercase mb-1">Period</div>
            <div className="text-sm text-[var(--signal-white)]">{modelInfo.trainingData.samples}</div>
          </div>
          <div>
            <div className="text-[10px] text-[var(--signal-dim)] uppercase mb-1">Features</div>
            <div className="text-sm text-[var(--signal-white)]">{modelInfo.trainingData.features} telemetry channels</div>
          </div>
          <div>
            <div className="text-[10px] text-[var(--signal-dim)] uppercase mb-1">Anomaly Ratio</div>
            <div className="text-sm text-[var(--signal-white)]">{modelInfo.trainingData.anomalyRatio}</div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="glass-panel p-6">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-5 h-5 text-[var(--accent-cyan)]" />
          <h3 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
            Performance Metrics
          </h3>
        </div>

        <div className="grid grid-cols-5 gap-4">
          {[
            { label: 'F1-Score', value: modelInfo.performance.f1Score, format: (v: number) => `${(v * 100).toFixed(1)}%` },
            { label: 'Precision', value: modelInfo.performance.precision, format: (v: number) => `${(v * 100).toFixed(0)}%` },
            { label: 'Recall', value: modelInfo.performance.recall, format: (v: number) => `${(v * 100).toFixed(0)}%` },
            { label: 'AUC-ROC', value: modelInfo.performance.aucRoc, format: (v: number) => `${(v * 100).toFixed(0)}%` },
            { label: 'FP Rate', value: modelInfo.performance.falsePositiveRate, format: (v: number) => `${(v * 100).toFixed(0)}%` },
          ].map((metric) => (
            <div key={metric.label} className="text-center">
              <div className="text-2xl font-mono font-bold text-[var(--signal-white)]">
                {metric.format(metric.value)}
              </div>
              <div className="text-[10px] text-[var(--signal-dim)] uppercase mt-1">{metric.label}</div>
              <div className="mt-2 h-1 bg-black/30 rounded-full overflow-hidden">
                <div
                  className="h-full bg-[var(--accent-cyan)]"
                  style={{ width: `${metric.value * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Known Limitations & Validity */}
      <div className="grid grid-cols-2 gap-4">
        {/* Blind Spots */}
        <div className="glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertCircle className="w-5 h-5 text-[var(--warning-amber)]" />
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
              Known Blind Spots
            </h3>
          </div>

          <ul className="space-y-2">
            {modelInfo.blindSpots.map((spot, i) => (
              <li key={i} className="flex items-start gap-2">
                <XCircle className="w-4 h-4 text-[var(--warning-amber)] mt-0.5 flex-shrink-0" />
                <span className="text-sm text-[var(--signal-dim)]">{spot}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Validity Conditions */}
        <div className="glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="w-5 h-5 text-[var(--nominal-green)]" />
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
              Validity Conditions
            </h3>
          </div>

          <ul className="space-y-2">
            {modelInfo.validityConditions.map((condition, i) => (
              <li key={i} className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-[var(--nominal-green)] mt-0.5 flex-shrink-0" />
                <span className="text-sm text-[var(--signal-dim)]">{condition}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="glass-panel p-4 border-l-2 border-[var(--warning-amber)]">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-[var(--warning-amber)] flex-shrink-0" />
          <div>
            <h4 className="text-sm font-semibold text-[var(--warning-amber)] mb-1">Model Disclaimer</h4>
            <p className="text-xs text-[var(--signal-dim)] leading-relaxed">
              This model is designed for anomaly detection assistance only. All critical decisions must be validated by qualified mission operators.
              The model's predictions should be considered alongside other telemetry sources and operational context.
              Performance metrics are based on historical validation data and may vary under novel operational conditions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
