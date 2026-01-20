// =============================================================================
// Globe Visualization - CSS/SVG Based (No Three.js dependency)
// =============================================================================
'use client';

import { useEffect, useRef, useState } from 'react';
import { useStore } from '@/store/useStore';

interface GlobeVizProps {
  className?: string;
}

export function GlobeViz({ className = '' }: GlobeVizProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [rotation, setRotation] = useState(0);

  const satellitePosition = useStore((s) => s.satellitePosition);
  const orbitTrail = useStore((s) => s.orbitTrail);
  const isAnomaly = useStore((s) => s.isAnomaly);
  const isPlaying = useStore((s) => s.isPlaying);

  // Auto-rotate globe
  useEffect(() => {
    if (!isPlaying) {
      const interval = setInterval(() => {
        setRotation((r) => (r + 0.2) % 360);
      }, 50);
      return () => clearInterval(interval);
    } else {
      // Follow satellite longitude
      setRotation(-satellitePosition.lon);
    }
  }, [isPlaying, satellitePosition.lon]);

  // Convert lat/lon to x/y on 2D projection
  const latLonToXY = (lat: number, lon: number, width: number, height: number) => {
    const adjustedLon = ((lon + rotation + 180) % 360) - 180;
    const x = ((adjustedLon + 180) / 360) * width;
    const y = ((90 - lat) / 180) * height;
    return { x, y, visible: Math.abs(adjustedLon) < 90 };
  };

  const width = 100;
  const height = 100;
  const satPos = latLonToXY(satellitePosition.lat, satellitePosition.lon, width, height);

  return (
    <div ref={containerRef} className={`relative w-full h-full overflow-hidden ${className}`}>
      {/* Stars background */}
      <div className="absolute inset-0 bg-[#0a0e14]">
        {Array.from({ length: 50 }).map((_, i) => (
          <div
            key={i}
            className="absolute w-px h-px bg-white rounded-full animate-pulse"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              opacity: Math.random() * 0.5 + 0.2,
              animationDelay: `${Math.random() * 2}s`,
            }}
          />
        ))}
      </div>

      {/* Globe container */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative" style={{ width: '70%', aspectRatio: '1' }}>
          {/* Earth sphere */}
          <svg viewBox="0 0 100 100" className="w-full h-full">
            <defs>
              {/* Earth gradient */}
              <radialGradient id="earthGradient" cx="30%" cy="30%">
                <stop offset="0%" stopColor="#4a90d9" />
                <stop offset="50%" stopColor="#1e3a5f" />
                <stop offset="100%" stopColor="#0d1b2a" />
              </radialGradient>

              {/* Atmosphere glow */}
              <radialGradient id="atmosphereGlow" cx="50%" cy="50%">
                <stop offset="85%" stopColor="transparent" />
                <stop offset="95%" stopColor={isAnomaly ? 'rgba(239, 68, 68, 0.3)' : 'rgba(34, 211, 238, 0.3)'} />
                <stop offset="100%" stopColor="transparent" />
              </radialGradient>

              {/* Grid pattern */}
              <pattern id="gridPattern" width="10" height="10" patternUnits="userSpaceOnUse">
                <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(34, 211, 238, 0.1)" strokeWidth="0.3" />
              </pattern>
            </defs>

            {/* Atmosphere glow */}
            <circle cx="50" cy="50" r="48" fill="url(#atmosphereGlow)" />

            {/* Earth */}
            <circle cx="50" cy="50" r="40" fill="url(#earthGradient)" />

            {/* Grid overlay */}
            <circle cx="50" cy="50" r="40" fill="url(#gridPattern)" opacity="0.5" />

            {/* Latitude lines */}
            {[-60, -30, 0, 30, 60].map((lat) => {
              const y = 50 + (lat / 90) * 40;
              const r = Math.cos((lat * Math.PI) / 180) * 40;
              return (
                <ellipse
                  key={lat}
                  cx="50"
                  cy={y}
                  rx={r}
                  ry={r * 0.3}
                  fill="none"
                  stroke="rgba(34, 211, 238, 0.15)"
                  strokeWidth="0.3"
                />
              );
            })}

            {/* Longitude lines */}
            {Array.from({ length: 12 }).map((_, i) => {
              const angle = (i * 30 + rotation) % 360;
              const rad = (angle * Math.PI) / 180;
              return (
                <ellipse
                  key={i}
                  cx="50"
                  cy="50"
                  rx={Math.abs(Math.sin(rad)) * 40}
                  ry="40"
                  fill="none"
                  stroke="rgba(34, 211, 238, 0.1)"
                  strokeWidth="0.3"
                  transform={`rotate(${0}, 50, 50)`}
                />
              );
            })}

            {/* Continents (simplified) */}
            <g opacity="0.4" transform={`rotate(${-rotation}, 50, 50)`}>
              {/* North America */}
              <ellipse cx="25" cy="35" rx="12" ry="8" fill="#2d5a3d" />
              {/* South America */}
              <ellipse cx="32" cy="60" rx="6" ry="10" fill="#2d5a3d" />
              {/* Europe/Africa */}
              <ellipse cx="52" cy="45" rx="8" ry="15" fill="#2d5a3d" />
              {/* Asia */}
              <ellipse cx="70" cy="35" rx="15" ry="10" fill="#2d5a3d" />
              {/* Australia */}
              <ellipse cx="78" cy="62" rx="6" ry="5" fill="#2d5a3d" />
            </g>

            {/* Orbit trail */}
            {orbitTrail.length > 1 && (
              <path
                d={orbitTrail
                  .map(([lon, lat], i) => {
                    const pos = latLonToXY(lat, lon, 80, 80);
                    const x = 10 + pos.x;
                    const y = 10 + pos.y;
                    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                  })
                  .join(' ')}
                fill="none"
                stroke="#22d3ee"
                strokeWidth="1"
                strokeDasharray="2,1"
                opacity="0.6"
              />
            )}

            {/* Satellite position */}
            {satPos.visible && (
              <g transform={`translate(${10 + satPos.x}, ${10 + satPos.y})`}>
                {/* Pulse effect */}
                <circle
                  r="6"
                  fill="none"
                  stroke={isAnomaly ? '#ef4444' : '#22d3ee'}
                  strokeWidth="0.5"
                  opacity="0.5"
                  className="animate-ping"
                />
                {/* Satellite dot */}
                <circle
                  r="2"
                  fill={isAnomaly ? '#ef4444' : '#22d3ee'}
                  className={isAnomaly ? 'animate-pulse' : ''}
                />
              </g>
            )}

            {/* Highlight rim */}
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke={isAnomaly ? 'rgba(239, 68, 68, 0.5)' : 'rgba(34, 211, 238, 0.3)'}
              strokeWidth="0.5"
            />
          </svg>

          {/* Orbital ring animation */}
          <div
            className="absolute inset-0 rounded-full border border-cyan-500/20"
            style={{
              transform: 'rotateX(70deg)',
              animation: 'spin 20s linear infinite',
            }}
          />
        </div>
      </div>

      {/* CSS for animations */}
      <style jsx>{`
        @keyframes spin {
          from { transform: rotateX(70deg) rotateZ(0deg); }
          to { transform: rotateX(70deg) rotateZ(360deg); }
        }
      `}</style>
    </div>
  );
}

export default GlobeViz;
