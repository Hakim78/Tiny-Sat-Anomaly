// =============================================================================
// ONNX Runtime Web - Session Management
// =============================================================================
// @ts-expect-error - onnxruntime-web types export issue
import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime - set wasm paths to public folder
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = '/';

// Singleton session instance
let session: ort.InferenceSession | null = null;
let isInitializing = false;
let initPromise: Promise<ort.InferenceSession> | null = null;

/**
 * Initialize ONNX Runtime session (singleton pattern)
 * Only loads the model once, subsequent calls return the cached session
 */
export async function initSession(): Promise<ort.InferenceSession> {
  // Return existing session
  if (session) {
    return session;
  }

  // Wait for ongoing initialization
  if (isInitializing && initPromise) {
    return initPromise;
  }

  // Start initialization
  isInitializing = true;

  initPromise = (async () => {
    try {
      console.log('[ONNX] Initializing inference session...');

      // Create session with WebAssembly backend
      const newSession = await ort.InferenceSession.create('/model.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      console.log('[ONNX] Session created successfully');
      console.log('[ONNX] Input names:', newSession.inputNames);
      console.log('[ONNX] Output names:', newSession.outputNames);

      session = newSession;
      return session;
    } catch (error) {
      console.error('[ONNX] Failed to create session:', error);
      throw error;
    } finally {
      isInitializing = false;
    }
  })();

  return initPromise;
}

/**
 * Run inference on a sequence window
 * @param window - 2D array of shape [50, 25] (window_size, features)
 * @returns Anomaly probability (0-1)
 */
export async function runInference(window: number[][]): Promise<number> {
  const sess = await initSession();

  // Validate input shape
  if (window.length !== 50 || window[0]?.length !== 25) {
    console.warn(
      `[ONNX] Invalid input shape: [${window.length}, ${window[0]?.length}]. Expected [50, 25]`
    );
    // Use statistical fallback for anomaly detection
    return computeStatisticalAnomaly(window);
  }

  try {
    // Flatten 2D array to 1D Float32Array
    const flatData = new Float32Array(50 * 25);
    for (let i = 0; i < 50; i++) {
      for (let j = 0; j < 25; j++) {
        flatData[i * 25 + j] = window[i][j];
      }
    }

    // Create tensor with shape [1, 50, 25] (batch, seq_len, features)
    const inputTensor = new ort.Tensor('float32', flatData, [1, 50, 25]);

    // Get input name from session
    const inputName = sess.inputNames[0] || 'input';

    // Run inference
    const results = await sess.run({ [inputName]: inputTensor });

    // Get output tensor (try multiple possible names)
    const outputName = sess.outputNames[0] || 'anomaly_probability';
    const output = results[outputName];

    if (output && output.data) {
      const probability = (output.data as Float32Array)[0];
      // If model returns very low values, blend with statistical detection
      if (probability < 0.01) {
        const statProb = computeStatisticalAnomaly(window);
        return Math.max(probability, statProb * 0.5);
      }
      return probability;
    }

    // Fallback to statistical anomaly detection
    return computeStatisticalAnomaly(window);
  } catch (error) {
    console.error('[ONNX] Inference error:', error);
    // Use statistical fallback
    return computeStatisticalAnomaly(window);
  }
}

/**
 * Statistical anomaly detection fallback
 * Uses z-score and variance analysis on the signal
 */
function computeStatisticalAnomaly(window: number[][]): number {
  if (!window || window.length === 0) return 0;

  // Extract first feature (primary signal)
  const signal = window.map((row) => row[0] || 0);

  // Compute mean and standard deviation
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const variance =
    signal.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) /
    signal.length;
  const std = Math.sqrt(variance);

  // Check last few values for anomaly (z-score > 2)
  const recentValues = signal.slice(-10);
  let anomalyScore = 0;

  for (const val of recentValues) {
    const zScore = std > 0 ? Math.abs(val - mean) / std : 0;
    if (zScore > 2) {
      anomalyScore += 0.1;
    } else if (zScore > 1.5) {
      anomalyScore += 0.05;
    }
  }

  // Also check for sudden jumps
  for (let i = 1; i < recentValues.length; i++) {
    const jump = Math.abs(recentValues[i] - recentValues[i - 1]);
    if (jump > std * 2) {
      anomalyScore += 0.15;
    }
  }

  // Clamp to 0-1
  return Math.min(1, Math.max(0, anomalyScore));
}

/**
 * Check if model is loaded
 */
export function isModelReady(): boolean {
  return session !== null;
}

/**
 * Dispose session (cleanup)
 */
export async function disposeSession(): Promise<void> {
  if (session) {
    await session.release();
    session = null;
  }
}
