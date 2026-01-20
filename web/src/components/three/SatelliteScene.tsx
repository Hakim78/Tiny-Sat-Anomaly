// =============================================================================
// Satellite Scene - Three.js Canvas with 3D Satellite
// =============================================================================
'use client';

import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Environment } from '@react-three/drei';
import { Satellite3D } from './Satellite3D';

interface SatelliteSceneProps {
  isAnomaly?: boolean;
  className?: string;
}

export function SatelliteScene({ isAnomaly = false, className = '' }: SatelliteSceneProps) {
  return (
    <div className={`w-full h-full ${className}`}>
      <Canvas
        camera={{ position: [5, 3, 5], fov: 45 }}
        style={{ background: 'transparent' }}
        gl={{ antialias: true, alpha: true }}
      >
        <Suspense fallback={null}>
          {/* Lighting */}
          <ambientLight intensity={0.3} />
          <directionalLight
            position={[10, 10, 5]}
            intensity={1.5}
            castShadow
            color="#ffffff"
          />
          <pointLight position={[-10, -10, -5]} intensity={0.5} color="#4488ff" />

          {/* Stars background */}
          <Stars
            radius={100}
            depth={50}
            count={3000}
            factor={4}
            saturation={0}
            fade
            speed={0.5}
          />

          {/* Satellite */}
          <Satellite3D isAnomaly={isAnomaly} scale={0.8} rotationSpeed={0.005} />

          {/* Controls */}
          <OrbitControls
            enablePan={false}
            enableZoom={true}
            minDistance={3}
            maxDistance={10}
            autoRotate={false}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}

export default SatelliteScene;
