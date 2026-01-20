// =============================================================================
// 3D Satellite Component - Loads GLB model
// =============================================================================
'use client';

import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import { Group, Mesh, MeshStandardMaterial } from 'three';

interface Satellite3DProps {
  position?: [number, number, number];
  scale?: number;
  isAnomaly?: boolean;
  rotationSpeed?: number;
}

export function Satellite3D({
  position = [0, 0, 0],
  scale = 1,
  isAnomaly = false,
  rotationSpeed = 0.005,
}: Satellite3DProps) {
  const groupRef = useRef<Group>(null);

  // Load the GLB model
  const { scene } = useGLTF('/models/satellite.glb');

  // Animate satellite rotation
  useFrame((state) => {
    if (groupRef.current) {
      // Slow rotation around Y axis
      groupRef.current.rotation.y += rotationSpeed;

      // Slight wobble for realism
      groupRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      groupRef.current.rotation.z = Math.cos(state.clock.elapsedTime * 0.3) * 0.05;
    }
  });

  // Clone the scene and apply anomaly color if needed
  const clonedScene = scene.clone();

  if (isAnomaly) {
    clonedScene.traverse((child) => {
      if (child instanceof Mesh && child.material instanceof MeshStandardMaterial) {
        child.material = child.material.clone();
        child.material.emissive.setHex(0xff0000);
        child.material.emissiveIntensity = 0.3;
      }
    });
  }

  return (
    <group ref={groupRef} position={position} scale={scale}>
      <primitive object={clonedScene} />

      {/* Status light */}
      <mesh position={[0, 1, 0]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshStandardMaterial
          color={isAnomaly ? '#ff0000' : '#00ff00'}
          emissive={isAnomaly ? '#ff0000' : '#00ff00'}
          emissiveIntensity={2}
        />
      </mesh>
    </group>
  );
}

// Preload the model
useGLTF.preload('/models/satellite.glb');

export default Satellite3D;
