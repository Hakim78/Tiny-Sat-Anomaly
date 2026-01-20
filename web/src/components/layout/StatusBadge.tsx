// =============================================================================
// Status Badge - NASA/SpaceX Style System Status Display
// =============================================================================
'use client';

import { useStore } from '@/store/useStore';
import {
  CheckCircle2,
  AlertTriangle,
  Pause,
  Loader2,
  Shield,
} from 'lucide-react';

interface StatusBadgeProps {
  className?: string;
}

export function StatusBadge({ className = '' }: StatusBadgeProps) {
  const isPlaying = useStore((s) => s.isPlaying);
  const isAnomaly = useStore((s) => s.isAnomaly);
  const isModelLoaded = useStore((s) => s.isModelLoaded);

  // Determine status
  let status: 'nominal' | 'critical' | 'standby' | 'loading' = 'standby';
  let statusText = 'STANDBY';
  let StatusIcon = Pause;

  if (!isModelLoaded) {
    status = 'loading';
    statusText = 'INITIALIZING';
    StatusIcon = Loader2;
  } else if (!isPlaying) {
    status = 'standby';
    statusText = 'STANDBY';
    StatusIcon = Pause;
  } else if (isAnomaly) {
    status = 'critical';
    statusText = 'CRITICAL';
    StatusIcon = AlertTriangle;
  } else {
    status = 'nominal';
    statusText = 'NOMINAL';
    StatusIcon = CheckCircle2;
  }

  const statusStyles = {
    nominal: {
      container: 'bg-green-500/10 border-green-500/30',
      text: 'text-green-400',
      icon: 'text-green-400',
      glow: 'shadow-green-500/20',
    },
    critical: {
      container: 'bg-red-500/10 border-red-500/30',
      text: 'text-red-400',
      icon: 'text-red-400 animate-pulse',
      glow: 'shadow-red-500/20',
    },
    standby: {
      container: 'bg-slate-800/50 border-slate-700',
      text: 'text-slate-400',
      icon: 'text-slate-500',
      glow: '',
    },
    loading: {
      container: 'bg-cyan-500/10 border-cyan-500/30',
      text: 'text-cyan-400',
      icon: 'text-cyan-400 animate-spin',
      glow: 'shadow-cyan-500/20',
    },
  };

  const styles = statusStyles[status];

  return (
    <div
      className={`rounded-lg border p-4 ${styles.container} ${styles.glow} ${className}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${status === 'critical' ? 'bg-red-500/20' : status === 'nominal' ? 'bg-green-500/20' : 'bg-slate-800'}`}>
            <StatusIcon className={`w-5 h-5 ${styles.icon}`} />
          </div>
          <div>
            <div className={`text-lg font-bold tracking-wider ${styles.text}`}>
              {statusText}
            </div>
            <div className="flex items-center gap-1.5 text-[10px] text-slate-500 uppercase tracking-wider">
              <Shield className="w-3 h-3" />
              System Status
            </div>
          </div>
        </div>

        {/* Status indicator dot */}
        <div className="flex flex-col items-center gap-1">
          <div
            className={`w-3 h-3 rounded-full ${
              status === 'nominal'
                ? 'bg-green-500'
                : status === 'critical'
                ? 'bg-red-500 animate-pulse'
                : status === 'loading'
                ? 'bg-cyan-500 animate-pulse'
                : 'bg-slate-600'
            }`}
          />
          <span className="text-[8px] text-slate-600 uppercase">Live</span>
        </div>
      </div>
    </div>
  );
}

export default StatusBadge;
