// =============================================================================
// Satellite Tracking Hook - Real-time Position Updates
// =============================================================================
import { useEffect, useRef, useCallback, useState } from 'react';
import { useStore } from '@/store/useStore';
import {
  getSatellitePosition,
  getSimulatedPosition,
  SMAP_NORAD_ID,
  type SatellitePosition,
} from '@/lib/satellite-api';

interface UseSatelliteTrackingOptions {
  apiKey?: string;
  noradId?: number;
  updateInterval?: number; // ms
  enabled?: boolean;
}

interface SatelliteTrackingState {
  isLive: boolean;
  isConnected: boolean;
  lastUpdate: Date | null;
  error: string | null;
  satelliteName: string;
}

export function useSatelliteTracking(options: UseSatelliteTrackingOptions = {}) {
  const {
    apiKey,
    noradId = SMAP_NORAD_ID,
    updateInterval = 5000, // 5 seconds default
    enabled = true,
  } = options;

  const [state, setState] = useState<SatelliteTrackingState>({
    isLive: false,
    isConnected: false,
    lastUpdate: null,
    error: null,
    satelliteName: 'SMAP',
  });

  const updateSatellitePosition = useStore((s) => s.updateSatellitePosition);
  const addLog = useStore((s) => s.addLog);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const hasLoggedConnection = useRef(false);

  // Fetch real satellite position
  const fetchRealPosition = useCallback(async () => {
    if (!apiKey) return null;

    try {
      const response = await getSatellitePosition(apiKey, noradId);

      if (response && response.positions.length > 0) {
        const pos = response.positions[0];

        setState((prev) => ({
          ...prev,
          isLive: true,
          isConnected: true,
          lastUpdate: new Date(),
          error: null,
          satelliteName: response.info.satname,
        }));

        // Log connection success once
        if (!hasLoggedConnection.current) {
          addLog('SUCCESS', `Connected to ${response.info.satname} (NORAD: ${noradId})`);
          hasLoggedConnection.current = true;
        }

        return {
          lat: pos.satlatitude,
          lon: pos.satlongitude,
          alt: pos.sataltitude,
        };
      }
    } catch (error) {
      setState((prev) => ({
        ...prev,
        isLive: false,
        isConnected: false,
        error: error instanceof Error ? error.message : 'Connection failed',
      }));
    }

    return null;
  }, [apiKey, noradId, addLog]);

  // Use simulated position as fallback
  const getSimulated = useCallback(() => {
    const simPos = getSimulatedPosition();
    return {
      lat: simPos.satlatitude,
      lon: simPos.satlongitude,
      alt: simPos.sataltitude,
    };
  }, []);

  // Main update function
  const updatePosition = useCallback(async () => {
    if (!enabled) return;

    let position: { lat: number; lon: number; alt: number } | null = null;

    // Try real API first if key is provided
    if (apiKey) {
      position = await fetchRealPosition();
    }

    // Fallback to simulation if no API key or API failed
    if (!position) {
      position = getSimulated();
      setState((prev) => ({
        ...prev,
        isLive: false,
        isConnected: !apiKey, // Not connected only if we tried API
        error: apiKey ? 'Using simulated data' : null,
      }));
    }

    // Update store
    if (position && updateSatellitePosition) {
      updateSatellitePosition(position);
    }
  }, [enabled, apiKey, fetchRealPosition, getSimulated, updateSatellitePosition]);

  // Start/stop tracking
  useEffect(() => {
    if (!enabled) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    // Initial update
    updatePosition();

    // Set up interval for periodic updates
    intervalRef.current = setInterval(updatePosition, updateInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [enabled, updateInterval, updatePosition]);

  // Manual refresh
  const refresh = useCallback(() => {
    updatePosition();
  }, [updatePosition]);

  return {
    ...state,
    refresh,
    noradId,
  };
}

// Environment variable helper
export function getN2YOApiKey(): string | undefined {
  if (typeof window !== 'undefined') {
    return process.env.NEXT_PUBLIC_N2YO_API_KEY;
  }
  return undefined;
}
