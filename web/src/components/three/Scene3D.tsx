// =============================================================================
// Scene3D - Professional 3D Scene with Realistic Earth & Satellite
// Uses @react-three/fiber and @react-three/drei for stable Three.js integration
// =============================================================================
'use client';

import { useRef, useMemo, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, OrbitControls, useGLTF, Float, useTexture } from '@react-three/drei';
import * as THREE from 'three';
import { useStore } from '@/store/useStore';
import { geoTo3D } from '@/lib/satellite-api';

// Scene constants - Increased Earth size for better visibility
const EARTH_RADIUS = 2.2;

// =============================================================================
// Earth with Texture Component (loads only main daymap)
// =============================================================================
function EarthWithTexture() {
  const earthRef = useRef<THREE.Mesh>(null);

  // Load only the main daymap texture (most important)
  const dayMap = useTexture('/textures/earth_daymap.jpg');

  // Configure texture
  useMemo(() => {
    dayMap.colorSpace = THREE.SRGBColorSpace;
    dayMap.anisotropy = 16;
  }, [dayMap]);

  // Rotate Earth slowly
  useFrame((_, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.015;
    }
  });

  // Atmosphere glow shader
  const atmosphereMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vec3 viewDir = normalize(-vPosition);
          float fresnel = pow(1.0 - abs(dot(vNormal, viewDir)), 3.0);
          vec3 atmosphereColor = vec3(0.3, 0.6, 1.0);
          gl_FragColor = vec4(atmosphereColor, fresnel * 0.5);
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
    });
  }, []);

  return (
    <group>
      {/* Main Earth with texture */}
      <mesh ref={earthRef}>
        <sphereGeometry args={[EARTH_RADIUS, 128, 128]} />
        <meshStandardMaterial
          map={dayMap}
          roughness={0.8}
          metalness={0.1}
        />
      </mesh>

      {/* Semi-transparent clouds layer */}
      <mesh rotation={[0, 0.5, 0]}>
        <sphereGeometry args={[EARTH_RADIUS + 0.015, 64, 64]} />
        <meshStandardMaterial
          color="#ffffff"
          transparent
          opacity={0.15}
          depthWrite={false}
        />
      </mesh>

      {/* Atmosphere glow */}
      <mesh scale={1.12}>
        <sphereGeometry args={[EARTH_RADIUS, 64, 64]} />
        <primitive object={atmosphereMaterial} attach="material" />
      </mesh>
    </group>
  );
}

// =============================================================================
// Fallback Earth (procedural, when texture fails to load)
// =============================================================================
function EarthFallback() {
  const earthRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.015;
    }
  });

  // Atmosphere glow
  const atmosphereMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vec3 viewDir = normalize(-vPosition);
          float fresnel = pow(1.0 - abs(dot(vNormal, viewDir)), 3.0);
          vec3 atmosphereColor = vec3(0.3, 0.6, 1.0);
          gl_FragColor = vec4(atmosphereColor, fresnel * 0.5);
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
    });
  }, []);

  return (
    <group>
      {/* Procedural Earth - Ocean */}
      <mesh ref={earthRef}>
        <sphereGeometry args={[EARTH_RADIUS, 64, 64]} />
        <meshStandardMaterial color="#1a4d7c" roughness={0.7} metalness={0.1} />
      </mesh>

      {/* Procedural continents overlay */}
      <mesh rotation={[0, 1, 0]}>
        <sphereGeometry args={[EARTH_RADIUS + 0.005, 64, 64]} />
        <meshStandardMaterial color="#2d5a3f" transparent opacity={0.5} roughness={0.9} />
      </mesh>

      {/* Atmosphere glow */}
      <mesh scale={1.12}>
        <sphereGeometry args={[EARTH_RADIUS, 64, 64]} />
        <primitive object={atmosphereMaterial} attach="material" />
      </mesh>
    </group>
  );
}

// =============================================================================
// Earth Component - tries texture, falls back to procedural
// =============================================================================
function Earth() {
  return (
    <Suspense fallback={<EarthFallback />}>
      <EarthWithTexture />
    </Suspense>
  );
}

// =============================================================================
// Satellite Component with Orbit Trail
// =============================================================================
function Satellite() {
  const satelliteRef = useRef<THREE.Group>(null);
  const trailRef = useRef<THREE.Line>(null);
  const isAnomaly = useStore((s) => s.isAnomaly);
  const satellitePosition = useStore((s) => s.satellitePosition);
  const orbitTrail = useStore((s) => s.orbitTrail);
  const isLiveTracking = useStore((s) => s.isLiveTracking);

  // Try to load the GLB model
  let gltf: ReturnType<typeof useGLTF> | null = null;
  try {
    gltf = useGLTF('/models/satellite.glb');
  } catch {
    // Model not found, will use fallback
  }

  // Satellite self-rotation animation
  useFrame((_, delta) => {
    if (satelliteRef.current) {
      satelliteRef.current.rotation.y += delta * 0.3;
      satelliteRef.current.rotation.x += delta * 0.1;
    }
  });

  // Convert geographic coordinates to 3D position
  const position3D = useMemo(() => {
    return geoTo3D(
      satellitePosition.lat,
      satellitePosition.lon,
      satellitePosition.alt,
      EARTH_RADIUS
    );
  }, [satellitePosition.lat, satellitePosition.lon, satellitePosition.alt]);

  // Calculate orbit radius for display ring
  const orbitRadius = useMemo(() => {
    return Math.sqrt(
      position3D.x * position3D.x + position3D.y * position3D.y + position3D.z * position3D.z
    );
  }, [position3D]);

  // Create orbit trail geometry from trail points
  const trailGeometry = useMemo(() => {
    if (orbitTrail.length < 2) return null;

    const points: THREE.Vector3[] = [];
    for (const [lon, lat] of orbitTrail) {
      const pos = geoTo3D(lat, lon, satellitePosition.alt, EARTH_RADIUS);
      points.push(new THREE.Vector3(pos.x, pos.y, pos.z));
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return geometry;
  }, [orbitTrail, satellitePosition.alt]);

  // Status color
  const statusColor = isAnomaly ? '#ff3b30' : isLiveTracking ? '#00ff88' : '#00d4ff';

  return (
    <group>
      {/* Orbit ring visualization */}
      <mesh rotation={[Math.PI / 2 + 0.17, 0, 0]}>
        <ringGeometry args={[orbitRadius - 0.015, orbitRadius + 0.015, 128]} />
        <meshBasicMaterial
          color={statusColor}
          transparent
          opacity={0.2}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Orbit trail */}
      {trailGeometry && (
        <line ref={trailRef}>
          <primitive object={trailGeometry} attach="geometry" />
          <lineBasicMaterial
            color={statusColor}
            transparent
            opacity={0.6}
            linewidth={2}
          />
        </line>
      )}

      {/* Satellite position */}
      <group position={[position3D.x, position3D.y, position3D.z]}>
        <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.2}>
          <group ref={satelliteRef} scale={0.06}>
            {gltf ? (
              <primitive object={gltf.scene.clone()} />
            ) : (
              // Professional fallback satellite geometry
              <group>
                {/* Main bus (body) */}
                <mesh>
                  <boxGeometry args={[1.2, 0.8, 1.8]} />
                  <meshStandardMaterial
                    color={isAnomaly ? '#cc3030' : '#e0e0e0'}
                    metalness={0.7}
                    roughness={0.3}
                  />
                </mesh>

                {/* Gold thermal insulation panels */}
                <mesh position={[0, 0.41, 0]}>
                  <boxGeometry args={[1.22, 0.02, 1.82]} />
                  <meshStandardMaterial
                    color="#d4a84b"
                    metalness={0.9}
                    roughness={0.1}
                  />
                </mesh>

                {/* Solar panel arm - left */}
                <mesh position={[-0.9, 0, 0]}>
                  <boxGeometry args={[0.6, 0.1, 0.2]} />
                  <meshStandardMaterial color="#888888" metalness={0.8} roughness={0.3} />
                </mesh>

                {/* Solar panel - left */}
                <mesh position={[-2.4, 0, 0]}>
                  <boxGeometry args={[2.4, 0.03, 1.4]} />
                  <meshStandardMaterial
                    color="#1a237e"
                    metalness={0.3}
                    roughness={0.4}
                  />
                </mesh>
                <mesh position={[-2.4, 0.02, 0]}>
                  <boxGeometry args={[2.3, 0.01, 1.3]} />
                  <meshStandardMaterial
                    color="#283593"
                    metalness={0.5}
                    roughness={0.2}
                    emissive="#1a237e"
                    emissiveIntensity={0.1}
                  />
                </mesh>

                {/* Solar panel arm - right */}
                <mesh position={[0.9, 0, 0]}>
                  <boxGeometry args={[0.6, 0.1, 0.2]} />
                  <meshStandardMaterial color="#888888" metalness={0.8} roughness={0.3} />
                </mesh>

                {/* Solar panel - right */}
                <mesh position={[2.4, 0, 0]}>
                  <boxGeometry args={[2.4, 0.03, 1.4]} />
                  <meshStandardMaterial
                    color="#1a237e"
                    metalness={0.3}
                    roughness={0.4}
                  />
                </mesh>
                <mesh position={[2.4, 0.02, 0]}>
                  <boxGeometry args={[2.3, 0.01, 1.3]} />
                  <meshStandardMaterial
                    color="#283593"
                    metalness={0.5}
                    roughness={0.2}
                    emissive="#1a237e"
                    emissiveIntensity={0.1}
                  />
                </mesh>

                {/* High-gain antenna dish */}
                <mesh position={[0, -0.6, 0.5]} rotation={[0.3, 0, 0]}>
                  <cylinderGeometry args={[0.5, 0.3, 0.15, 32]} />
                  <meshStandardMaterial color="#f5f5f5" metalness={0.6} roughness={0.3} />
                </mesh>
                <mesh position={[0, -0.75, 0.55]} rotation={[0.3, 0, 0]}>
                  <cylinderGeometry args={[0.05, 0.05, 0.4]} />
                  <meshStandardMaterial color="#666666" metalness={0.9} roughness={0.2} />
                </mesh>

                {/* Status beacon */}
                <mesh position={[0, 0.55, 0]}>
                  <sphereGeometry args={[0.1, 16, 16]} />
                  <meshStandardMaterial
                    color={statusColor}
                    emissive={statusColor}
                    emissiveIntensity={isAnomaly ? 2 : 1}
                  />
                </mesh>

                {/* Sensor array */}
                <mesh position={[0, -0.42, -0.7]}>
                  <boxGeometry args={[0.4, 0.04, 0.4]} />
                  <meshStandardMaterial color="#333333" metalness={0.8} roughness={0.4} />
                </mesh>
              </group>
            )}
          </group>
        </Float>

        {/* Point light for satellite glow */}
        <pointLight
          color={statusColor}
          intensity={isAnomaly ? 3 : 1.5}
          distance={1.5}
        />

        {/* Small sprite for visibility at distance */}
        <sprite scale={[0.15, 0.15, 1]}>
          <spriteMaterial
            color={statusColor}
            transparent
            opacity={0.8}
            depthTest={false}
          />
        </sprite>
      </group>
    </group>
  );
}

// =============================================================================
// Loading Fallback
// =============================================================================
function LoadingFallback() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += delta;
      meshRef.current.rotation.y += delta * 0.5;
    }
  });

  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[0.5, 1]} />
      <meshBasicMaterial color="#00ffff" wireframe />
    </mesh>
  );
}

// =============================================================================
// Main Scene3D Component
// =============================================================================
interface Scene3DProps {
  className?: string;
}

export function Scene3D({ className = '' }: Scene3DProps) {
  const isAnomaly = useStore((s) => s.isAnomaly);

  return (
    <div className={`relative w-full h-full ${className}`}>
      <Canvas
        camera={{ position: [0, 2, 6.5], fov: 50 }}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance',
        }}
        style={{ background: 'transparent', width: '100%', height: '100%' }}
        dpr={[1, 2]}
      >
        {/* Professional lighting setup for realistic Earth */}
        <ambientLight intensity={0.4} color="#ffffff" />

        {/* Sun light - main directional (bright to show textures) */}
        <directionalLight
          position={[5, 3, 4]}
          intensity={3}
          color="#fff5e0"
        />

        {/* Fill light - subtle blue from space side */}
        <directionalLight
          position={[-4, -1, -3]}
          intensity={0.8}
          color="#87ceeb"
        />

        {/* Top rim light for definition */}
        <directionalLight
          position={[0, 5, 0]}
          intensity={0.5}
          color="#ffffff"
        />

        {/* Hemisphere light for natural ambient */}
        <hemisphereLight
          args={['#87ceeb', '#1a1a2e', 0.3]}
        />

        {/* Starfield background */}
        <Stars
          radius={100}
          depth={50}
          count={7000}
          factor={4}
          saturation={0}
          fade
          speed={0.5}
        />

        {/* Main scene elements */}
        <Suspense fallback={<LoadingFallback />}>
          <Earth />
          <Satellite />
        </Suspense>

        {/* Camera controls */}
        <OrbitControls
          enableZoom={true}
          enablePan={false}
          minDistance={4}
          maxDistance={18}
          autoRotate
          autoRotateSpeed={0.2}
          maxPolarAngle={Math.PI * 0.85}
          minPolarAngle={Math.PI * 0.15}
          enableDamping
          dampingFactor={0.05}
          makeDefault
        />

        {/* Alert effect when anomaly */}
        {isAnomaly && (
          <mesh scale={50}>
            <sphereGeometry args={[1, 8, 8]} />
            <meshBasicMaterial
              color="#ff3b30"
              transparent
              opacity={0.02}
              side={THREE.BackSide}
            />
          </mesh>
        )}
      </Canvas>

      {/* HUD corner brackets */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-0 left-0 w-8 h-8 border-l-2 border-t-2 border-cyan-500/50" />
        <div className="absolute top-0 right-0 w-8 h-8 border-r-2 border-t-2 border-cyan-500/50" />
        <div className="absolute bottom-0 left-0 w-8 h-8 border-l-2 border-b-2 border-cyan-500/50" />
        <div className="absolute bottom-0 right-0 w-8 h-8 border-r-2 border-b-2 border-cyan-500/50" />
      </div>
    </div>
  );
}

// Preload satellite model
useGLTF.preload('/models/satellite.glb');
