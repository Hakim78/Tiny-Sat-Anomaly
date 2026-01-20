// =============================================================================
// Control Panel - NASA/SpaceX Style Sidebar Controls
// =============================================================================
'use client';

import { useStore } from '@/store/useStore';
import { useInference } from '@/hooks/useInference';
import {
  Settings,
  Package,
  Play,
  Square,
  RotateCcw,
  Zap,
  Target,
  Flame,
  BarChart3,
  CheckCircle2,
  XCircle,
  Loader2,
  AlertCircle,
} from 'lucide-react';

interface ControlPanelProps {
  className?: string;
}

export function ControlPanel({ className = '' }: ControlPanelProps) {
  const { isModelLoaded, isLoading, error } = useInference();

  const isPlaying = useStore((s) => s.isPlaying);
  const isDataLoaded = useStore((s) => s.isDataLoaded);
  const togglePlayback = useStore((s) => s.togglePlayback);
  const reset = useStore((s) => s.reset);
  const playbackSpeed = useStore((s) => s.playbackSpeed);
  const setPlaybackSpeed = useStore((s) => s.setPlaybackSpeed);
  const anomalyThreshold = useStore((s) => s.anomalyThreshold);
  const setAnomalyThreshold = useStore((s) => s.setAnomalyThreshold);
  const triggerSabotage = useStore((s) => s.triggerSabotage);
  const isSabotageActive = useStore((s) => s.isSabotageActive);
  const currentIndex = useStore((s) => s.currentIndex);
  const anomalyCount = useStore((s) => s.anomalyCount);
  const telemetryData = useStore((s) => s.telemetryData);

  const totalSamples = telemetryData?.metadata.samples || 0;
  const progress = totalSamples > 0 ? (currentIndex / totalSamples) * 100 : 0;

  const canStart = isModelLoaded && isDataLoaded && !isLoading;

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
          <Settings className="w-4 h-4 text-cyan-400" />
        </div>
        <div>
          <h2 className="text-sm font-bold text-white tracking-wider uppercase">
            Mission Control
          </h2>
          <div className="h-px w-full bg-gradient-to-r from-cyan-500/50 to-transparent mt-1" />
        </div>
      </div>

      {/* Resource Status */}
      <div className="rounded-lg bg-slate-900/50 border border-slate-800 p-3 space-y-2">
        <div className="flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-wider">
          <Package className="w-3 h-3" />
          System Resources
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-xs">
            {isLoading ? (
              <Loader2 className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
            ) : isModelLoaded ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
            ) : (
              <XCircle className="w-3.5 h-3.5 text-slate-600" />
            )}
            <span className={isModelLoaded ? 'text-green-400' : 'text-slate-500'}>
              {isLoading ? 'Loading ONNX...' : isModelLoaded ? 'Model Online' : 'Model Offline'}
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            {isDataLoaded ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
            ) : (
              <XCircle className="w-3.5 h-3.5 text-slate-600" />
            )}
            <span className={isDataLoaded ? 'text-green-400' : 'text-slate-500'}>
              {isDataLoaded ? `Telemetry: ${totalSamples.toLocaleString()} PKT` : 'No Telemetry'}
            </span>
          </div>
          {error && (
            <div className="flex items-center gap-2 text-xs text-red-400">
              <AlertCircle className="w-3.5 h-3.5" />
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Playback Controls */}
      <div className="space-y-2">
        <button
          onClick={togglePlayback}
          disabled={!canStart}
          className={`w-full py-3 rounded-lg font-bold uppercase tracking-wider text-sm transition-all flex items-center justify-center gap-2 ${
            canStart
              ? isPlaying
                ? 'bg-red-500/20 border border-red-500/50 text-red-400 hover:bg-red-500/30'
                : 'bg-cyan-500/20 border border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/30'
              : 'bg-slate-800 text-slate-600 cursor-not-allowed border border-slate-700'
          }`}
        >
          {isPlaying ? (
            <>
              <Square className="w-4 h-4" />
              Abort Mission
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Start Simulation
            </>
          )}
        </button>

        <button
          onClick={reset}
          className="w-full py-2 rounded-lg font-semibold text-xs uppercase tracking-wider bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white transition-all flex items-center justify-center gap-2 border border-slate-700"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </button>
      </div>

      {/* Speed Control */}
      <div className="rounded-lg bg-slate-900/50 border border-slate-800 p-3 space-y-2">
        <div className="flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-wider">
          <Zap className="w-3 h-3" />
          Playback Speed
        </div>
        <div className="flex gap-1.5">
          {[1, 2, 4, 8].map((speed) => (
            <button
              key={speed}
              onClick={() => setPlaybackSpeed(speed)}
              className={`flex-1 py-1.5 rounded text-xs font-mono transition-all ${
                playbackSpeed === speed
                  ? 'bg-cyan-500 text-slate-900 font-bold'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>

      {/* Threshold Control */}
      <div className="rounded-lg bg-slate-900/50 border border-slate-800 p-3 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-wider">
            <Target className="w-3 h-3" />
            Alert Threshold
          </div>
          <span className="text-cyan-400 font-mono text-xs font-bold">
            {(anomalyThreshold * 100).toFixed(0)}%
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={anomalyThreshold}
          onChange={(e) => setAnomalyThreshold(parseFloat(e.target.value))}
          className="w-full h-1.5 bg-slate-700 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-cyan-400"
        />
      </div>

      {/* Chaos Engineering */}
      <div className="rounded-lg bg-slate-900/50 border border-slate-800 p-3 space-y-2">
        <div className="flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-wider">
          <Flame className="w-3 h-3" />
          Chaos Engineering
        </div>
        <button
          onClick={triggerSabotage}
          disabled={!isPlaying || isSabotageActive}
          className={`w-full py-2 rounded-lg font-semibold text-xs uppercase tracking-wider transition-all flex items-center justify-center gap-2 ${
            isPlaying && !isSabotageActive
              ? 'bg-gradient-to-r from-orange-600/80 to-red-600/80 text-white hover:from-orange-500 hover:to-red-500 border border-orange-500/30'
              : 'bg-slate-800 text-slate-600 cursor-not-allowed border border-slate-700'
          }`}
        >
          <Flame className="w-3.5 h-3.5" />
          Trigger Solar Flare
        </button>
        {isSabotageActive && (
          <div className="text-[10px] text-orange-400 animate-pulse text-center flex items-center justify-center gap-1">
            <Flame className="w-3 h-3" />
            Solar flare active!
          </div>
        )}
      </div>

      {/* Mission Stats */}
      <div className="rounded-lg bg-slate-900/50 border border-slate-800 p-3 space-y-3">
        <div className="flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-wider">
          <BarChart3 className="w-3 h-3" />
          Mission Statistics
        </div>

        {/* Progress bar */}
        <div className="space-y-1">
          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="text-[10px] text-slate-500 text-right font-mono">
            {progress.toFixed(1)}% Complete
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="bg-slate-800/50 rounded-lg p-2 text-center">
            <div className="text-lg font-bold text-cyan-400 font-mono">
              {currentIndex.toLocaleString()}
            </div>
            <div className="text-[9px] text-slate-500 uppercase">Packets</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-2 text-center">
            <div className={`text-lg font-bold font-mono ${anomalyCount > 0 ? 'text-red-400' : 'text-green-400'}`}>
              {anomalyCount}
            </div>
            <div className="text-[9px] text-slate-500 uppercase">Anomalies</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ControlPanel;
