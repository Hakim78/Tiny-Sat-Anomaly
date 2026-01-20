// =============================================================================
// Enhanced Logs Component - Professional Event Logging with Traceability
// =============================================================================
'use client';

import { useState, useMemo } from 'react';
import { useStore } from '@/store/useStore';
import {
  ScrollText,
  Filter,
  Download,
  Search,
  AlertTriangle,
  CheckCircle,
  Info,
  AlertCircle,
  Link2,
  Clock,
  ChevronDown,
} from 'lucide-react';

interface EnhancedLogEntry {
  eventId: string;
  timestamp: Date;
  type: 'INFO' | 'WARN' | 'ERROR' | 'SUCCESS';
  message: string;
  correlationId?: string;
  subsystem?: string;
  severity?: number;
}

export function EnhancedLogs() {
  const logs = useStore((s) => s.logs);
  const [filter, setFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedLog, setExpandedLog] = useState<string | null>(null);

  // Generate enhanced log entries with event IDs and correlation
  const enhancedLogs = useMemo<EnhancedLogEntry[]>(() => {
    let correlationCounter = 1;
    let lastAnomalyCorrelation: string | null = null;

    return logs.map((log, index) => {
      // Generate event ID
      const dateStr = log.timestamp.toISOString().split('T')[0].replace(/-/g, '');
      const eventId = `EVT-${dateStr}-${String(index + 1).padStart(4, '0')}`;

      // Determine subsystem based on message content
      let subsystem = 'SYSTEM';
      if (log.message.toLowerCase().includes('power')) subsystem = 'PWR';
      else if (log.message.toLowerCase().includes('thermal')) subsystem = 'THM';
      else if (log.message.toLowerCase().includes('telemetry')) subsystem = 'TLM';
      else if (log.message.toLowerCase().includes('anomaly')) subsystem = 'ADX';
      else if (log.message.toLowerCase().includes('solar')) subsystem = 'ENV';
      else if (log.message.toLowerCase().includes('mission')) subsystem = 'MSN';

      // Assign correlation IDs for related events
      let correlationId: string | undefined;
      if (log.type === 'ERROR' || log.message.toLowerCase().includes('anomaly')) {
        correlationCounter++;
        lastAnomalyCorrelation = `COR-${String(correlationCounter).padStart(3, '0')}`;
        correlationId = lastAnomalyCorrelation;
      } else if (lastAnomalyCorrelation && (log.type === 'WARN' || log.type === 'SUCCESS')) {
        correlationId = lastAnomalyCorrelation;
        if (log.type === 'SUCCESS') {
          lastAnomalyCorrelation = null;
        }
      }

      // Calculate severity
      let severity = 1;
      if (log.type === 'WARN') severity = 2;
      if (log.type === 'ERROR') severity = 3;
      if (log.message.includes('CRITICAL')) severity = 4;

      return {
        eventId,
        timestamp: log.timestamp,
        type: log.type,
        message: log.message,
        correlationId,
        subsystem,
        severity,
      };
    });
  }, [logs]);

  // Filter logs
  const filteredLogs = useMemo(() => {
    return enhancedLogs.filter((log) => {
      if (filter !== 'all' && log.type !== filter) return false;
      if (searchQuery && !log.message.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !log.eventId.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    });
  }, [enhancedLogs, filter, searchQuery]);

  // Statistics
  const stats = useMemo(() => {
    return {
      total: enhancedLogs.length,
      errors: enhancedLogs.filter((l) => l.type === 'ERROR').length,
      warnings: enhancedLogs.filter((l) => l.type === 'WARN').length,
      info: enhancedLogs.filter((l) => l.type === 'INFO').length,
      success: enhancedLogs.filter((l) => l.type === 'SUCCESS').length,
    };
  }, [enhancedLogs]);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'ERROR': return <AlertCircle className="w-4 h-4 text-[var(--alert-red)]" />;
      case 'WARN': return <AlertTriangle className="w-4 h-4 text-[var(--warning-amber)]" />;
      case 'SUCCESS': return <CheckCircle className="w-4 h-4 text-[var(--nominal-green)]" />;
      default: return <Info className="w-4 h-4 text-[var(--accent-cyan)]" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'ERROR': return 'text-[var(--alert-red)]';
      case 'WARN': return 'text-[var(--warning-amber)]';
      case 'SUCCESS': return 'text-[var(--nominal-green)]';
      default: return 'text-[var(--accent-cyan)]';
    }
  };

  const exportLogs = () => {
    const logText = filteredLogs.map((log) =>
      `[${log.timestamp.toISOString()}] [${log.eventId}] [${log.type}] ${log.correlationId ? `[${log.correlationId}] ` : ''}${log.message}`
    ).join('\n');

    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mission-logs-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
  };

  return (
    <div className="space-y-4">
      {/* Stats Bar */}
      <div className="grid grid-cols-5 gap-3">
        {[
          { label: 'Total Events', value: stats.total, color: 'text-[var(--signal-white)]' },
          { label: 'Errors', value: stats.errors, color: 'text-[var(--alert-red)]' },
          { label: 'Warnings', value: stats.warnings, color: 'text-[var(--warning-amber)]' },
          { label: 'Info', value: stats.info, color: 'text-[var(--accent-cyan)]' },
          { label: 'Success', value: stats.success, color: 'text-[var(--nominal-green)]' },
        ].map((stat) => (
          <div key={stat.label} className="glass-panel p-3 text-center">
            <div className={`text-2xl font-mono font-bold ${stat.color}`}>{stat.value}</div>
            <div className="text-[9px] text-[var(--signal-dim)] uppercase">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="glass-panel p-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <ScrollText className="w-5 h-5 text-[var(--accent-cyan)]" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
              Event Logs
            </h2>
          </div>

          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--signal-dim)]" />
              <input
                type="text"
                placeholder="Search events..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 pr-4 py-2 text-xs bg-black/30 border border-white/10 rounded text-[var(--signal-white)] placeholder:text-[var(--signal-dim)] focus:outline-none focus:border-[var(--accent-cyan)]/50 w-48"
              />
            </div>

            {/* Filter */}
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--signal-dim)]" />
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="pl-9 pr-8 py-2 text-xs bg-black/30 border border-white/10 rounded text-[var(--signal-white)] focus:outline-none focus:border-[var(--accent-cyan)]/50 appearance-none cursor-pointer"
              >
                <option value="all">All Types</option>
                <option value="ERROR">Errors</option>
                <option value="WARN">Warnings</option>
                <option value="INFO">Info</option>
                <option value="SUCCESS">Success</option>
              </select>
              <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--signal-dim)] pointer-events-none" />
            </div>

            {/* Export */}
            <button
              onClick={exportLogs}
              className="flex items-center gap-2 px-3 py-2 text-xs bg-black/30 border border-white/10 rounded text-[var(--signal-dim)] hover:text-[var(--signal-white)] hover:border-white/20 transition-colors"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Log Entries */}
      <div className="glass-panel p-4">
        <div className="space-y-1 max-h-[600px] overflow-y-auto">
          {filteredLogs.length === 0 ? (
            <div className="text-center py-8 text-[var(--signal-dim)]">
              <ScrollText className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No events to display</p>
            </div>
          ) : (
            filteredLogs.map((log) => (
              <div
                key={log.eventId}
                className={`group border border-white/5 rounded-lg overflow-hidden transition-colors hover:border-white/10 ${
                  expandedLog === log.eventId ? 'bg-white/5' : ''
                }`}
              >
                <button
                  onClick={() => setExpandedLog(expandedLog === log.eventId ? null : log.eventId)}
                  className="w-full px-3 py-2 flex items-center gap-3 text-left"
                >
                  {/* Icon */}
                  {getTypeIcon(log.type)}

                  {/* Event ID */}
                  <span className="text-[10px] font-mono text-[var(--signal-dim)] w-28 flex-shrink-0">
                    {log.eventId}
                  </span>

                  {/* Timestamp */}
                  <span className="text-[10px] font-mono text-[var(--signal-dim)] w-20 flex-shrink-0">
                    {log.timestamp.toLocaleTimeString('en-US', { hour12: false })}
                  </span>

                  {/* Subsystem */}
                  <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-black/30 text-[var(--accent-cyan)] w-12 text-center flex-shrink-0">
                    {log.subsystem}
                  </span>

                  {/* Message */}
                  <span className={`text-xs flex-1 truncate ${getTypeColor(log.type)}`}>
                    {log.message}
                  </span>

                  {/* Correlation ID */}
                  {log.correlationId && (
                    <span className="flex items-center gap-1 text-[9px] text-[var(--signal-dim)]">
                      <Link2 className="w-3 h-3" />
                      {log.correlationId}
                    </span>
                  )}
                </button>

                {/* Expanded Details */}
                {expandedLog === log.eventId && (
                  <div className="px-4 py-3 border-t border-white/5 bg-black/20">
                    <div className="grid grid-cols-4 gap-4 text-[10px]">
                      <div>
                        <div className="text-[var(--signal-dim)] mb-1">Full Timestamp</div>
                        <div className="text-[var(--signal-white)] font-mono">
                          {log.timestamp.toISOString()}
                        </div>
                      </div>
                      <div>
                        <div className="text-[var(--signal-dim)] mb-1">Event Type</div>
                        <div className={`font-semibold ${getTypeColor(log.type)}`}>
                          {log.type}
                        </div>
                      </div>
                      <div>
                        <div className="text-[var(--signal-dim)] mb-1">Subsystem</div>
                        <div className="text-[var(--signal-white)]">{log.subsystem}</div>
                      </div>
                      <div>
                        <div className="text-[var(--signal-dim)] mb-1">Severity Level</div>
                        <div className="text-[var(--signal-white)]">{log.severity}/4</div>
                      </div>
                    </div>

                    {log.correlationId && (
                      <div className="mt-3 pt-3 border-t border-white/5">
                        <div className="text-[var(--signal-dim)] text-[10px] mb-2">Correlated Events</div>
                        <div className="flex flex-wrap gap-2">
                          {enhancedLogs
                            .filter((l) => l.correlationId === log.correlationId && l.eventId !== log.eventId)
                            .slice(0, 5)
                            .map((related) => (
                              <button
                                key={related.eventId}
                                onClick={() => setExpandedLog(related.eventId)}
                                className="text-[9px] font-mono px-2 py-1 rounded bg-black/30 text-[var(--accent-cyan)] hover:bg-black/50 transition-colors"
                              >
                                {related.eventId}
                              </button>
                            ))}
                        </div>
                      </div>
                    )}

                    <div className="mt-3 pt-3 border-t border-white/5">
                      <div className="text-[var(--signal-dim)] text-[10px] mb-1">Full Message</div>
                      <div className="text-xs text-[var(--signal-white)] bg-black/30 rounded p-2 font-mono">
                        {log.message}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
