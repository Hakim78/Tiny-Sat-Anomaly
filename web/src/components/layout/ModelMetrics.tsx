// =============================================================================
// ModelMetrics - ML Model Performance Display
// Shows trained model metrics for professional presentation
// =============================================================================
'use client';

import { Brain, Target, TrendingUp, BarChart3, Layers, Clock } from 'lucide-react';

interface ModelMetricsProps {
  compact?: boolean;
}

// Model performance metrics from training
const MODEL_METRICS = {
  architecture: 'LSTM Autoencoder',
  f1Score: 0.5346,
  precision: 0.42,
  recall: 0.73,
  auc: 0.78,
  inputShape: '[50, 25]',
  windowSize: 50,
  features: 25,
  threshold: 0.5,
  trainingSamples: 12000,
  framework: 'TensorFlow/Keras → ONNX',
  inference: 'ONNX Runtime WebAssembly',
};

export function ModelMetrics({ compact = false }: ModelMetricsProps) {
  if (compact) {
    return (
      <div className="glass-panel p-3">
        <div className="flex items-center gap-2 mb-2">
          <Brain className="w-4 h-4 text-[var(--accent-cyan)]" />
          <span className="text-[10px] text-[var(--signal-dim)] uppercase tracking-wider">
            Model Performance
          </span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <div className="text-center">
            <div className="text-lg font-mono font-bold text-[var(--accent-cyan)]">
              {(MODEL_METRICS.f1Score * 100).toFixed(1)}%
            </div>
            <div className="text-[8px] text-[var(--signal-dim)] uppercase">F1-Score</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-mono font-bold text-[var(--nominal-green)]">
              {(MODEL_METRICS.recall * 100).toFixed(0)}%
            </div>
            <div className="text-[8px] text-[var(--signal-dim)] uppercase">Recall</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-mono font-bold text-[var(--signal-white)]">
              {(MODEL_METRICS.auc * 100).toFixed(0)}%
            </div>
            <div className="text-[8px] text-[var(--signal-dim)] uppercase">AUC-ROC</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel-accent p-4">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-lg bg-[var(--accent-purple)]/10 flex items-center justify-center">
          <Brain className="w-5 h-5 text-[var(--accent-purple)]" />
        </div>
        <div>
          <div className="text-sm font-semibold">Neural Network Model</div>
          <div className="text-[9px] text-[var(--signal-dim)] uppercase tracking-wider">
            {MODEL_METRICS.architecture}
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="space-y-3">
        <div className="text-[9px] text-[var(--signal-dim)] uppercase tracking-wider mb-2">
          Performance Metrics
        </div>

        {/* F1-Score with progress bar */}
        <div>
          <div className="flex justify-between items-center mb-1">
            <div className="flex items-center gap-2">
              <Target className="w-3 h-3 text-[var(--accent-cyan)]" />
              <span className="text-[10px] text-[var(--signal-grey)]">F1-Score</span>
            </div>
            <span className="text-sm font-mono font-semibold text-[var(--accent-cyan)]">
              {(MODEL_METRICS.f1Score * 100).toFixed(2)}%
            </span>
          </div>
          <div className="progress-bar">
            <div
              className="fill"
              style={{ width: `${MODEL_METRICS.f1Score * 100}%` }}
            />
          </div>
        </div>

        {/* Precision & Recall */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-black/30 rounded-lg p-2">
            <div className="flex items-center gap-1 mb-1">
              <TrendingUp className="w-3 h-3 text-[var(--signal-grey)]" />
              <span className="text-[8px] text-[var(--signal-dim)] uppercase">Precision</span>
            </div>
            <div className="text-sm font-mono text-[var(--signal-white)]">
              {(MODEL_METRICS.precision * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-black/30 rounded-lg p-2">
            <div className="flex items-center gap-1 mb-1">
              <BarChart3 className="w-3 h-3 text-[var(--nominal-green)]" />
              <span className="text-[8px] text-[var(--signal-dim)] uppercase">Recall</span>
            </div>
            <div className="text-sm font-mono text-[var(--nominal-green)]">
              {(MODEL_METRICS.recall * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* AUC-ROC */}
        <div className="bg-black/30 rounded-lg p-2">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-1">
              <BarChart3 className="w-3 h-3 text-[var(--accent-purple)]" />
              <span className="text-[8px] text-[var(--signal-dim)] uppercase">AUC-ROC</span>
            </div>
            <span className="text-sm font-mono text-[var(--accent-purple)]">
              {(MODEL_METRICS.auc * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Model Architecture */}
      <div className="mt-4 pt-3 border-t border-white/5">
        <div className="text-[9px] text-[var(--signal-dim)] uppercase tracking-wider mb-2">
          Architecture
        </div>
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div className="flex items-center gap-2">
            <Layers className="w-3 h-3 text-[var(--signal-grey)]" />
            <span className="text-[var(--signal-dim)]">Input:</span>
            <span className="font-mono text-[var(--signal-white)]">{MODEL_METRICS.inputShape}</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="w-3 h-3 text-[var(--signal-grey)]" />
            <span className="text-[var(--signal-dim)]">Window:</span>
            <span className="font-mono text-[var(--signal-white)]">{MODEL_METRICS.windowSize} steps</span>
          </div>
        </div>
        <div className="mt-2 text-[9px] text-[var(--signal-dim)]">
          <span className="text-[var(--accent-cyan)]">{MODEL_METRICS.framework}</span>
          <span className="mx-2">→</span>
          <span>{MODEL_METRICS.inference}</span>
        </div>
      </div>
    </div>
  );
}
