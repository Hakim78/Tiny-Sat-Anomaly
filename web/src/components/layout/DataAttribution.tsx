// =============================================================================
// DataAttribution - Data Sources & Credits for Professional Presentation
// =============================================================================
'use client';

import { Database, Globe, Cpu, ExternalLink, Award } from 'lucide-react';

interface DataAttributionProps {
  expanded?: boolean;
}

const DATA_SOURCES = [
  {
    name: 'NASA SMAP Mission',
    description: 'Soil Moisture Active Passive satellite telemetry',
    url: 'https://smap.jpl.nasa.gov/',
    icon: Globe,
    color: 'var(--accent-cyan)',
  },
  {
    name: 'Kaggle Dataset',
    description: 'Satellite anomaly detection training data',
    url: 'https://www.kaggle.com/datasets',
    icon: Database,
    color: 'var(--accent-purple)',
  },
  {
    name: 'ONNX Runtime',
    description: 'Cross-platform ML inference engine',
    url: 'https://onnxruntime.ai/',
    icon: Cpu,
    color: 'var(--nominal-green)',
  },
];

const TECHNOLOGIES = [
  { name: 'Next.js 14', category: 'Framework' },
  { name: 'React Three Fiber', category: '3D Rendering' },
  { name: 'TensorFlow/Keras', category: 'Model Training' },
  { name: 'ONNX WebAssembly', category: 'Browser Inference' },
  { name: 'Zustand', category: 'State Management' },
  { name: 'Tailwind CSS', category: 'Styling' },
];

export function DataAttribution({ expanded = false }: DataAttributionProps) {
  if (!expanded) {
    return (
      <div className="glass-panel p-3">
        <div className="flex items-center gap-2 mb-2">
          <Award className="w-4 h-4 text-[var(--accent-cyan)]" />
          <span className="text-[10px] text-[var(--signal-dim)] uppercase tracking-wider">
            Data Sources
          </span>
        </div>
        <div className="flex flex-wrap gap-1">
          {DATA_SOURCES.map((source) => (
            <a
              key={source.name}
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 px-2 py-1 rounded bg-black/30 hover:bg-black/50 transition-colors text-[9px] text-[var(--signal-grey)] hover:text-[var(--signal-white)]"
            >
              <source.icon className="w-3 h-3" style={{ color: source.color }} />
              {source.name}
            </a>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel-accent p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-[var(--accent-cyan)]/10 flex items-center justify-center">
          <Award className="w-5 h-5 text-[var(--accent-cyan)]" />
        </div>
        <div>
          <div className="text-sm font-semibold">Data Sources & Attribution</div>
          <div className="text-[9px] text-[var(--signal-dim)] uppercase tracking-wider">
            Academic & Research Credits
          </div>
        </div>
      </div>

      {/* Data Sources */}
      <div>
        <div className="text-[9px] text-[var(--signal-dim)] uppercase tracking-wider mb-2">
          Primary Data Sources
        </div>
        <div className="space-y-2">
          {DATA_SOURCES.map((source) => (
            <a
              key={source.name}
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 p-2 rounded-lg bg-black/20 hover:bg-black/40 transition-colors group"
            >
              <div
                className="w-8 h-8 rounded flex items-center justify-center"
                style={{ backgroundColor: `${source.color}15` }}
              >
                <source.icon className="w-4 h-4" style={{ color: source.color }} />
              </div>
              <div className="flex-1">
                <div className="text-xs font-medium text-[var(--signal-white)] group-hover:text-[var(--accent-cyan)] transition-colors">
                  {source.name}
                </div>
                <div className="text-[9px] text-[var(--signal-dim)]">
                  {source.description}
                </div>
              </div>
              <ExternalLink className="w-3 h-3 text-[var(--signal-dim)] group-hover:text-[var(--accent-cyan)] transition-colors" />
            </a>
          ))}
        </div>
      </div>

      {/* Technologies */}
      <div className="pt-3 border-t border-white/5">
        <div className="text-[9px] text-[var(--signal-dim)] uppercase tracking-wider mb-2">
          Technology Stack
        </div>
        <div className="flex flex-wrap gap-1">
          {TECHNOLOGIES.map((tech) => (
            <div
              key={tech.name}
              className="px-2 py-1 rounded bg-black/30 text-[9px]"
              title={tech.category}
            >
              <span className="text-[var(--signal-grey)]">{tech.name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Citation */}
      <div className="pt-3 border-t border-white/5">
        <div className="text-[9px] text-[var(--signal-dim)] uppercase tracking-wider mb-2">
          Citation
        </div>
        <div className="bg-black/30 rounded p-2 font-mono text-[9px] text-[var(--signal-grey)]">
          TINY-SAT Anomaly Detection System (2024). LSTM Autoencoder for
          Satellite Telemetry Analysis. IPSSI MIA4 Project.
        </div>
      </div>
    </div>
  );
}
