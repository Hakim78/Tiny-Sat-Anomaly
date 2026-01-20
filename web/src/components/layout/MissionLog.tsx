// =============================================================================
// Mission Log - NASA/SpaceX Style Terminal Log
// =============================================================================
'use client';

import { useStore, LogEntry } from '@/store/useStore';
import {
  Terminal,
  Info,
  AlertTriangle,
  XCircle,
  CheckCircle2,
} from 'lucide-react';

interface MissionLogProps {
  className?: string;
  maxEntries?: number;
}

export function MissionLog({ className = '', maxEntries = 10 }: MissionLogProps) {
  const logs = useStore((s) => s.logs);

  const typeConfig: Record<LogEntry['type'], { icon: typeof Info; color: string }> = {
    INFO: { icon: Info, color: 'text-cyan-400' },
    WARN: { icon: AlertTriangle, color: 'text-yellow-400' },
    ERROR: { icon: XCircle, color: 'text-red-400' },
    SUCCESS: { icon: CheckCircle2, color: 'text-green-400' },
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  return (
    <div className={`rounded-lg bg-slate-900/50 border border-slate-800 p-4 h-full flex flex-col ${className}`}>
      <div className="flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-wider mb-3">
        <Terminal className="w-3 h-3" />
        Mission Log
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin space-y-1 font-mono text-xs">
        {logs.length === 0 ? (
          <div className="flex items-center gap-2 text-slate-600 py-1">
            <span className="text-slate-700">[--:--:--]</span>
            <Info className="w-3 h-3" />
            <span>Awaiting mission start...</span>
          </div>
        ) : (
          logs.slice(0, maxEntries).map((log) => {
            const config = typeConfig[log.type];
            const Icon = config.icon;

            return (
              <div
                key={log.id}
                className="flex items-start gap-2 py-1 border-b border-slate-800/50 last:border-0"
              >
                <span className="text-slate-600 shrink-0">
                  [{formatTime(log.timestamp)}]
                </span>
                <Icon className={`w-3 h-3 shrink-0 mt-0.5 ${config.color}`} />
                <span className="text-slate-300 break-all">{log.message}</span>
              </div>
            );
          })
        )}
      </div>

      {/* Terminal cursor effect */}
      <div className="mt-2 pt-2 border-t border-slate-800">
        <div className="flex items-center gap-2 text-slate-600">
          <span className="text-cyan-500">$</span>
          <span className="animate-pulse">_</span>
        </div>
      </div>
    </div>
  );
}

export default MissionLog;
