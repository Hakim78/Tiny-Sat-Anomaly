// =============================================================================
// useInference Hook - ONNX Inference Logic
// =============================================================================
'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { initSession, runInference, isModelReady } from '@/lib/onnx';
import { useStore } from '@/store/useStore';

interface UseInferenceReturn {
  isModelLoaded: boolean;
  isLoading: boolean;
  error: string | null;
  runPrediction: () => Promise<void>;
}

/**
 * Hook for managing ONNX model inference
 * Handles model loading and prediction execution
 */
export function useInference(): UseInferenceReturn {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const setModelLoaded = useStore((s) => s.setModelLoaded);
  const updateInferenceResult = useStore((s) => s.updateInferenceResult);
  const getCurrentWindow = useStore((s) => s.getCurrentWindow);
  const isModelLoaded = useStore((s) => s.isModelLoaded);

  // Initialize model on mount
  useEffect(() => {
    let mounted = true;

    async function loadModel() {
      try {
        setIsLoading(true);
        setError(null);

        await initSession();

        if (mounted) {
          setModelLoaded(true);
          setIsLoading(false);
          console.log('[useInference] Model loaded successfully');
        }
      } catch (err) {
        if (mounted) {
          const message = err instanceof Error ? err.message : 'Failed to load model';
          setError(message);
          setIsLoading(false);
          console.error('[useInference] Model loading failed:', err);
        }
      }
    }

    loadModel();

    return () => {
      mounted = false;
    };
  }, [setModelLoaded]);

  // Run prediction on current window
  const runPrediction = useCallback(async () => {
    if (!isModelReady()) {
      console.warn('[useInference] Model not ready');
      return;
    }

    const window = getCurrentWindow();
    if (!window) {
      // Not enough data for a full window yet
      return;
    }

    try {
      const probability = await runInference(window);
      updateInferenceResult(probability);
    } catch (err) {
      console.error('[useInference] Prediction failed:', err);
    }
  }, [getCurrentWindow, updateInferenceResult]);

  return {
    isModelLoaded,
    isLoading,
    error,
    runPrediction,
  };
}

/**
 * Hook for the simulation game loop
 * Manages timing and inference execution
 */
export function useSimulationLoop() {
  const isPlaying = useStore((s) => s.isPlaying);
  const playbackSpeed = useStore((s) => s.playbackSpeed);
  const tick = useStore((s) => s.tick);
  const isModelLoaded = useStore((s) => s.isModelLoaded);

  const { runPrediction } = useInference();
  const frameRef = useRef<number>();
  const lastTickRef = useRef<number>(0);

  useEffect(() => {
    if (!isPlaying || !isModelLoaded) {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      return;
    }

    // Base interval: 100ms at 1x speed
    const baseInterval = 100;
    const interval = baseInterval / playbackSpeed;

    const gameLoop = (timestamp: number) => {
      if (timestamp - lastTickRef.current >= interval) {
        lastTickRef.current = timestamp;

        // Advance simulation
        tick();

        // Run inference
        runPrediction();
      }

      frameRef.current = requestAnimationFrame(gameLoop);
    };

    frameRef.current = requestAnimationFrame(gameLoop);

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, tick, runPrediction, isModelLoaded]);
}
