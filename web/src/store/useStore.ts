// =============================================================================
// Zustand Store - Global Application State
// =============================================================================
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// Types
export interface TelemetryData {
  metadata: {
    source: string;
    samples: number;
    features: number;
    normalized: boolean;
  };
  data: number[][];
}

export interface LogEntry {
  id: string;
  timestamp: Date;
  type: 'INFO' | 'WARN' | 'ERROR' | 'SUCCESS';
  message: string;
}

export interface SatellitePosition {
  lat: number;
  lon: number;
  alt: number;
}

interface SimulationState {
  // Data
  telemetryData: TelemetryData | null;
  isDataLoaded: boolean;
  isModelLoaded: boolean;

  // Playback
  isPlaying: boolean;
  currentIndex: number;
  playbackSpeed: number; // 1x, 2x, 4x, etc.

  // Inference results
  anomalyProbability: number;
  isAnomaly: boolean;
  anomalyThreshold: number;
  anomalyCount: number;

  // History for charts (sliding window)
  signalHistory: number[];
  probabilityHistory: number[];
  anomalyIndices: number[];

  // Satellite position
  satellitePosition: SatellitePosition;
  orbitTrail: [number, number][]; // [lon, lat][]
  isLiveTracking: boolean;

  // Mission log
  logs: LogEntry[];

  // Sabotage mode (Chaos Engineering)
  isSabotageActive: boolean;
  sabotageCountdown: number;

  // Actions
  setTelemetryData: (data: TelemetryData) => void;
  setModelLoaded: (loaded: boolean) => void;
  togglePlayback: () => void;
  setPlaybackSpeed: (speed: number) => void;
  setAnomalyThreshold: (threshold: number) => void;
  tick: () => void;
  reset: () => void;
  updateInferenceResult: (probability: number) => void;
  triggerSabotage: () => void;
  addLog: (type: LogEntry['type'], message: string) => void;
  getCurrentWindow: () => number[][] | null;
  updateSatellitePosition: (position: SatellitePosition) => void;
  setLiveTracking: (isLive: boolean) => void;
}

// Constants
const WINDOW_SIZE = 50;
const HISTORY_SIZE = 150;
const ORBIT_TRAIL_SIZE = 60;

// Helper: Calculate satellite position (polar orbit simulation)
function calculateOrbitPosition(t: number): SatellitePosition {
  const lat = Math.sin(t * 0.02) * 75; // Oscillate between ±75°
  const lon = ((t * 0.8) % 360) - 180; // Rotate around Earth
  const alt = 685 + Math.sin(t * 0.1) * 20; // ~685km with variation
  return { lat, lon, alt };
}

// Generate unique ID for logs
const generateId = () => Math.random().toString(36).substr(2, 9);

// Create store with selector middleware for optimized subscriptions
export const useStore = create<SimulationState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    telemetryData: null,
    isDataLoaded: false,
    isModelLoaded: false,

    isPlaying: false,
    currentIndex: 0,
    playbackSpeed: 1,

    anomalyProbability: 0,
    isAnomaly: false,
    anomalyThreshold: 0.3, // 30% threshold - more sensitive for demos
    anomalyCount: 0,

    signalHistory: [],
    probabilityHistory: [],
    anomalyIndices: [],

    satellitePosition: { lat: 0, lon: 0, alt: 685 },
    orbitTrail: [],
    isLiveTracking: false,

    logs: [],

    isSabotageActive: false,
    sabotageCountdown: 0,

    // Actions
    setTelemetryData: (data) =>
      set({
        telemetryData: data,
        isDataLoaded: true,
      }),

    setModelLoaded: (loaded) =>
      set({ isModelLoaded: loaded }),

    togglePlayback: () =>
      set((state) => {
        const newPlaying = !state.isPlaying;
        if (newPlaying && state.currentIndex === 0) {
          // Starting fresh - add log
          const newLog: LogEntry = {
            id: generateId(),
            timestamp: new Date(),
            type: 'INFO' as const,
            message: 'Mission simulation started',
          };
          return {
            isPlaying: newPlaying,
            logs: [newLog, ...state.logs].slice(0, 50),
          };
        }
        return { isPlaying: newPlaying };
      }),

    setPlaybackSpeed: (speed) =>
      set({ playbackSpeed: speed }),

    setAnomalyThreshold: (threshold) =>
      set({ anomalyThreshold: threshold }),

    tick: () =>
      set((state) => {
        if (!state.isPlaying || !state.telemetryData) return state;

        const { data } = state.telemetryData;
        const nextIndex = state.currentIndex + 1;

        // Check if simulation complete
        if (nextIndex >= data.length) {
          const completeLog: LogEntry = {
            id: generateId(),
            timestamp: new Date(),
            type: 'SUCCESS' as const,
            message: `Mission complete! Processed ${data.length} samples. Detected ${state.anomalyCount} anomalies.`,
          };
          return {
            isPlaying: false,
            logs: [completeLog, ...state.logs].slice(0, 50),
          };
        }

        // Get current sample
        let currentSample = data[nextIndex];

        // Apply sabotage if active
        let sabotageCountdown = state.sabotageCountdown;
        if (sabotageCountdown > 0) {
          // Inject noise into the signal
          currentSample = currentSample.map(
            (v) => v + (Math.random() - 0.3) * 2
          );
          sabotageCountdown--;

          if (sabotageCountdown === 0) {
            const stabilizeLog: LogEntry = {
              id: generateId(),
              timestamp: new Date(),
              type: 'SUCCESS' as const,
              message: 'Solar flare subsided - systems stabilizing',
            };
            return {
              ...state,
              currentIndex: nextIndex,
              sabotageCountdown: 0,
              isSabotageActive: false,
              signalHistory: [
                ...state.signalHistory.slice(-HISTORY_SIZE + 1),
                currentSample[0],
              ],
              satellitePosition: calculateOrbitPosition(nextIndex),
              orbitTrail: [
                ...state.orbitTrail.slice(-ORBIT_TRAIL_SIZE + 1),
                [
                  calculateOrbitPosition(nextIndex).lon,
                  calculateOrbitPosition(nextIndex).lat,
                ] as [number, number],
              ],
              logs: [stabilizeLog, ...state.logs].slice(0, 50),
            };
          }
        }

        // Update signal history
        const newSignalHistory = [
          ...state.signalHistory.slice(-HISTORY_SIZE + 1),
          currentSample[0],
        ];

        // Update satellite position
        const newPosition = calculateOrbitPosition(nextIndex);
        const newTrail: [number, number][] = [
          ...state.orbitTrail.slice(-ORBIT_TRAIL_SIZE + 1),
          [newPosition.lon, newPosition.lat],
        ];

        return {
          currentIndex: nextIndex,
          signalHistory: newSignalHistory,
          satellitePosition: newPosition,
          orbitTrail: newTrail,
          sabotageCountdown,
          isSabotageActive: sabotageCountdown > 0,
        };
      }),

    reset: () =>
      set({
        isPlaying: false,
        currentIndex: 0,
        anomalyProbability: 0,
        isAnomaly: false,
        anomalyCount: 0,
        signalHistory: [],
        probabilityHistory: [],
        anomalyIndices: [],
        orbitTrail: [],
        satellitePosition: { lat: 0, lon: 0, alt: 685 },
        logs: [
          {
            id: generateId(),
            timestamp: new Date(),
            type: 'INFO' as const,
            message: 'System reset - ready for new mission',
          },
        ],
        isSabotageActive: false,
        sabotageCountdown: 0,
      }),

    updateInferenceResult: (probability) =>
      set((state) => {
        const isAnomaly = probability > state.anomalyThreshold;
        const newProbHistory = [
          ...state.probabilityHistory.slice(-HISTORY_SIZE + 1),
          probability * 100,
        ];

        let newAnomalyIndices = state.anomalyIndices;
        let newAnomalyCount = state.anomalyCount;
        let newLogs = state.logs;

        if (isAnomaly) {
          newAnomalyIndices = [
            ...state.anomalyIndices,
            state.signalHistory.length - 1,
          ];
          newAnomalyCount = state.anomalyCount + 1;
          const anomalyLog: LogEntry = {
            id: generateId(),
            timestamp: new Date(),
            type: 'ERROR' as const,
            message: `ANOMALY DETECTED @ step ${state.currentIndex} (P=${(probability * 100).toFixed(1)}%)`,
          };
          newLogs = [anomalyLog, ...state.logs].slice(0, 50);
        }

        return {
          anomalyProbability: probability,
          isAnomaly,
          anomalyCount: newAnomalyCount,
          probabilityHistory: newProbHistory,
          anomalyIndices: newAnomalyIndices,
          logs: newLogs,
        };
      }),

    triggerSabotage: () =>
      set((state) => {
        const sabotageLog: LogEntry = {
          id: generateId(),
          timestamp: new Date(),
          type: 'WARN' as const,
          message: 'SOLAR FLARE INCOMING - Expect signal degradation!',
        };
        return {
          isSabotageActive: true,
          sabotageCountdown: 15, // 15 ticks of chaos
          logs: [sabotageLog, ...state.logs].slice(0, 50),
        };
      }),

    addLog: (type, message) =>
      set((state) => ({
        logs: [
          {
            id: generateId(),
            timestamp: new Date(),
            type,
            message,
          },
          ...state.logs,
        ].slice(0, 50),
      })),

    getCurrentWindow: () => {
      const state = get();
      if (!state.telemetryData || state.currentIndex < WINDOW_SIZE - 1) {
        return null;
      }

      const startIdx = state.currentIndex - WINDOW_SIZE + 1;
      let window = state.telemetryData.data.slice(startIdx, state.currentIndex + 1);

      // Apply sabotage noise if active
      if (state.isSabotageActive) {
        window = window.map((row) =>
          row.map((v) => v + (Math.random() - 0.3) * 2)
        );
      }

      return window;
    },

    updateSatellitePosition: (position) =>
      set((state) => {
        const newTrail: [number, number][] = [
          ...state.orbitTrail.slice(-ORBIT_TRAIL_SIZE + 1),
          [position.lon, position.lat],
        ];
        return {
          satellitePosition: position,
          orbitTrail: newTrail,
        };
      }),

    setLiveTracking: (isLive) =>
      set({ isLiveTracking: isLive }),
  }))
);

// Selector hooks for optimized re-renders
export const useIsPlaying = () => useStore((s) => s.isPlaying);
export const useCurrentIndex = () => useStore((s) => s.currentIndex);
export const useAnomalyProbability = () => useStore((s) => s.anomalyProbability);
export const useIsAnomaly = () => useStore((s) => s.isAnomaly);
export const useSatellitePosition = () => useStore((s) => s.satellitePosition);
export const useSignalHistory = () => useStore((s) => s.signalHistory);
export const useLogs = () => useStore((s) => s.logs);
export const useIsLiveTracking = () => useStore((s) => s.isLiveTracking);
