// =============================================================================
// SOP Panel Component - Standard Operating Procedures for Anomaly Response
// =============================================================================
'use client';

import { useState, useEffect } from 'react';
import { useStore } from '@/store/useStore';
import {
  Shield,
  CheckCircle2,
  Circle,
  AlertTriangle,
  ChevronRight,
  Clock,
  User,
  FileText,
} from 'lucide-react';

interface SOPStep {
  id: number;
  title: string;
  description: string;
  completed: boolean;
  required: boolean;
}

interface SOP {
  id: string;
  code: string;
  title: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  triggerCondition: string;
  steps: SOPStep[];
  estimatedTime: string;
  lastExecuted: string | null;
}

const standardProcedures: SOP[] = [
  {
    id: 'sop-a17',
    code: 'SOP-A17',
    title: 'Power Subsystem Anomaly',
    severity: 'high',
    triggerCondition: 'Power fluctuation > 10% baseline',
    steps: [
      { id: 1, title: 'Verify telemetry source', description: 'Confirm data integrity from power monitoring unit', completed: false, required: true },
      { id: 2, title: 'Check solar array status', description: 'Review panel orientation and illumination', completed: false, required: true },
      { id: 3, title: 'Validate thermal readings', description: 'Cross-reference with thermal subsystem data', completed: false, required: true },
      { id: 4, title: 'Assess battery state', description: 'Check charge level and discharge rate', completed: false, required: true },
      { id: 5, title: 'Evaluate load distribution', description: 'Review power consumption by subsystem', completed: false, required: false },
      { id: 6, title: 'Document findings', description: 'Log observations and recommended actions', completed: false, required: true },
    ],
    estimatedTime: '15-20 min',
    lastExecuted: null,
  },
  {
    id: 'sop-t09',
    code: 'SOP-T09',
    title: 'Thermal Excursion Response',
    severity: 'medium',
    triggerCondition: 'Temperature drift > 5°C from nominal',
    steps: [
      { id: 1, title: 'Identify affected component', description: 'Locate thermal anomaly source', completed: false, required: true },
      { id: 2, title: 'Review heater status', description: 'Check active thermal control elements', completed: false, required: true },
      { id: 3, title: 'Assess orbital position', description: 'Correlate with eclipse/sun exposure', completed: false, required: true },
      { id: 4, title: 'Initiate mitigation', description: 'Adjust thermal control if necessary', completed: false, required: false },
    ],
    estimatedTime: '10-15 min',
    lastExecuted: null, // Dynamically calculated
  },
  {
    id: 'sop-c03',
    code: 'SOP-C03',
    title: 'Communication Degradation',
    severity: 'low',
    triggerCondition: 'Signal-to-noise ratio < threshold',
    steps: [
      { id: 1, title: 'Check antenna pointing', description: 'Verify ground station lock', completed: false, required: true },
      { id: 2, title: 'Review link budget', description: 'Assess current link margin', completed: false, required: true },
      { id: 3, title: 'Consider interference', description: 'Check for RF environment issues', completed: false, required: false },
    ],
    estimatedTime: '5-10 min',
    lastExecuted: null, // Dynamically calculated
  },
];

export function SOPPanel() {
  const isAnomaly = useStore((s) => s.isAnomaly);
  const anomalyProbability = useStore((s) => s.anomalyProbability);
  const addLog = useStore((s) => s.addLog);

  const [activeSOP, setActiveSOP] = useState<string | null>(null);
  const [completedSteps, setCompletedSteps] = useState<Record<string, number[]>>({});
  const [sessionStart] = useState(() => new Date());
  const [sessionDuration, setSessionDuration] = useState('0h 0m');

  // Update session duration every minute
  useEffect(() => {
    const updateDuration = () => {
      const now = new Date();
      const diff = Math.floor((now.getTime() - sessionStart.getTime()) / 1000);
      const hours = Math.floor(diff / 3600);
      const minutes = Math.floor((diff % 3600) / 60);
      setSessionDuration(`${hours}h ${minutes}m`);
    };
    updateDuration();
    const interval = setInterval(updateDuration, 60000);
    return () => clearInterval(interval);
  }, [sessionStart]);

  // Determine which SOP to suggest based on anomaly state
  const suggestedSOP = isAnomaly ? standardProcedures[0] : null;

  const toggleStep = (sopId: string, stepId: number) => {
    setCompletedSteps((prev) => {
      const current = prev[sopId] || [];
      if (current.includes(stepId)) {
        return { ...prev, [sopId]: current.filter((id) => id !== stepId) };
      } else {
        addLog('INFO', `SOP step completed: ${sopId} - Step ${stepId}`);
        return { ...prev, [sopId]: [...current, stepId] };
      }
    });
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-[var(--alert-red)] bg-[var(--alert-red)]/10 border-[var(--alert-red)]/30';
      case 'high': return 'text-[var(--warning-amber)] bg-[var(--warning-amber)]/10 border-[var(--warning-amber)]/30';
      case 'medium': return 'text-[var(--accent-cyan)] bg-[var(--accent-cyan)]/10 border-[var(--accent-cyan)]/30';
      default: return 'text-[var(--nominal-green)] bg-[var(--nominal-green)]/10 border-[var(--nominal-green)]/30';
    }
  };

  const getStepProgress = (sop: SOP) => {
    const completed = completedSteps[sop.id]?.length || 0;
    const required = sop.steps.filter((s) => s.required).length;
    return { completed, total: sop.steps.length, required };
  };

  return (
    <div className="space-y-4">
      {/* Active Alert Banner */}
      {suggestedSOP && (
        <div className="glass-panel p-4 border-l-4 border-[var(--alert-red)] animate-pulse-slow">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-[var(--alert-red)]" />
              <div>
                <div className="text-sm font-semibold text-[var(--alert-red)]">
                  Anomaly Detected - Procedure Required
                </div>
                <div className="text-xs text-[var(--signal-dim)]">
                  Recommended: {suggestedSOP.code} - {suggestedSOP.title}
                </div>
              </div>
            </div>
            <button
              onClick={() => setActiveSOP(suggestedSOP.id)}
              className="px-4 py-2 text-xs font-semibold rounded bg-[var(--alert-red)]/20 text-[var(--alert-red)] border border-[var(--alert-red)]/30 hover:bg-[var(--alert-red)]/30 transition-colors"
            >
              INITIATE SOP
            </button>
          </div>
        </div>
      )}

      {/* SOP List */}
      <div className="glass-panel p-6">
        <div className="flex items-center gap-2 mb-6">
          <Shield className="w-5 h-5 text-[var(--accent-cyan)]" />
          <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--signal-white)]">
            Standard Operating Procedures
          </h2>
        </div>

        <div className="space-y-3">
          {standardProcedures.map((sop) => {
            const isActive = activeSOP === sop.id;
            const progress = getStepProgress(sop);

            return (
              <div key={sop.id} className="border border-white/10 rounded-lg overflow-hidden">
                {/* SOP Header */}
                <button
                  onClick={() => setActiveSOP(isActive ? null : sop.id)}
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className={`px-2 py-0.5 text-[9px] font-bold rounded border ${getSeverityColor(sop.severity)}`}>
                      {sop.severity.toUpperCase()}
                    </span>
                    <div className="text-left">
                      <div className="text-sm font-semibold text-[var(--signal-white)]">
                        <span className="text-[var(--accent-cyan)] font-mono mr-2">{sop.code}</span>
                        {sop.title}
                      </div>
                      <div className="text-[10px] text-[var(--signal-dim)]">
                        Trigger: {sop.triggerCondition}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    {progress.completed > 0 && (
                      <div className="text-[10px] text-[var(--signal-dim)]">
                        {progress.completed}/{progress.total} steps
                      </div>
                    )}
                    <ChevronRight className={`w-4 h-4 text-[var(--signal-dim)] transition-transform ${isActive ? 'rotate-90' : ''}`} />
                  </div>
                </button>

                {/* SOP Steps */}
                {isActive && (
                  <div className="border-t border-white/10 bg-black/20 p-4">
                    <div className="flex items-center gap-4 mb-4 text-[10px] text-[var(--signal-dim)]">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Est. time: {sop.estimatedTime}
                      </div>
                      {sop.lastExecuted && (
                        <div className="flex items-center gap-1">
                          <FileText className="w-3 h-3" />
                          Last run: {sop.lastExecuted}
                        </div>
                      )}
                    </div>

                    <div className="space-y-2">
                      {sop.steps.map((step) => {
                        const isCompleted = completedSteps[sop.id]?.includes(step.id);

                        return (
                          <button
                            key={step.id}
                            onClick={() => toggleStep(sop.id, step.id)}
                            className={`w-full flex items-start gap-3 p-3 rounded-lg transition-colors ${
                              isCompleted
                                ? 'bg-[var(--nominal-green)]/10 border border-[var(--nominal-green)]/30'
                                : 'bg-black/20 border border-white/5 hover:border-white/10'
                            }`}
                          >
                            {isCompleted ? (
                              <CheckCircle2 className="w-5 h-5 text-[var(--nominal-green)] flex-shrink-0" />
                            ) : (
                              <Circle className="w-5 h-5 text-[var(--signal-dim)] flex-shrink-0" />
                            )}
                            <div className="text-left flex-1">
                              <div className={`text-sm ${isCompleted ? 'text-[var(--nominal-green)]' : 'text-[var(--signal-white)]'}`}>
                                <span className="font-mono text-[var(--signal-dim)] mr-2">
                                  {step.id}.
                                </span>
                                {step.title}
                                {step.required && (
                                  <span className="ml-2 text-[8px] text-[var(--warning-amber)]">REQUIRED</span>
                                )}
                              </div>
                              <div className="text-[10px] text-[var(--signal-dim)] mt-0.5">
                                {step.description}
                              </div>
                            </div>
                          </button>
                        );
                      })}
                    </div>

                    {/* Completion Status */}
                    <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-between">
                      <div className="text-[10px] text-[var(--signal-dim)]">
                        Required steps: {completedSteps[sop.id]?.filter((id) =>
                          sop.steps.find((s) => s.id === id)?.required
                        ).length || 0}/{progress.required}
                      </div>
                      {progress.completed >= progress.required && (
                        <button
                          onClick={() => {
                            addLog('SUCCESS', `SOP ${sop.code} completed by operator`);
                            setActiveSOP(null);
                          }}
                          className="px-4 py-2 text-xs font-semibold rounded bg-[var(--nominal-green)]/20 text-[var(--nominal-green)] border border-[var(--nominal-green)]/30 hover:bg-[var(--nominal-green)]/30 transition-colors"
                        >
                          MARK COMPLETE
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Operator Session Info */}
      <div className="glass-panel p-4">
        <div className="flex items-center gap-2 mb-3">
          <User className="w-4 h-4 text-[var(--accent-cyan)]" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--signal-white)]">
            Operator Session
          </h3>
        </div>

        <div className="grid grid-cols-3 gap-4 text-[10px]">
          <div>
            <div className="text-[var(--signal-dim)]">Operator ID</div>
            <div className="text-[var(--signal-white)] font-mono">MCO-L2-OPERATOR</div>
          </div>
          <div>
            <div className="text-[var(--signal-dim)]">Shift Start</div>
            <div className="text-[var(--signal-white)] font-mono">
              {sessionStart.toISOString().replace('T', ' ').slice(0, 19)} UTC
            </div>
          </div>
          <div>
            <div className="text-[var(--signal-dim)]">Session Duration</div>
            <div className="text-[var(--signal-white)] font-mono">{sessionDuration}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
