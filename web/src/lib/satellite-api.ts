// =============================================================================
// N2YO Satellite Tracking API - Full Integration
// =============================================================================
// Documentation: https://www.n2yo.com/api/
// Free tier limits: TLE: 1000/hr, Positions: 1000/hr, Passes: 100/hr
// =============================================================================

// -----------------------------------------------------------------------------
// Type Definitions
// -----------------------------------------------------------------------------

export interface SatelliteInfo {
  satid: number;
  satname: string;
  transactionscount: number;
}

export interface SatellitePosition {
  satlatitude: number;
  satlongitude: number;
  sataltitude: number;
  azimuth: number;
  elevation: number;
  ra: number;
  dec: number;
  timestamp: number;
}

export interface TLEResponse {
  info: SatelliteInfo;
  tle: string;
}

export interface PositionsResponse {
  info: SatelliteInfo;
  positions: SatellitePosition[];
}

export interface VisualPass {
  startAz: number;
  startAzCompass: string;
  startEl: number;
  startUTC: number;
  maxAz: number;
  maxAzCompass: string;
  maxEl: number;
  maxUTC: number;
  endAz: number;
  endAzCompass: string;
  endEl: number;
  endUTC: number;
  mag: number;
  duration: number;
}

export interface VisualPassesResponse {
  info: SatelliteInfo & { passescount: number };
  passes: VisualPass[];
}

export interface RadioPass {
  startAz: number;
  startAzCompass: string;
  startUTC: number;
  maxAz: number;
  maxAzCompass: string;
  maxEl: number;
  maxUTC: number;
  endAz: number;
  endAzCompass: string;
  endUTC: number;
}

export interface RadioPassesResponse {
  info: SatelliteInfo & { passescount: number };
  passes: RadioPass[];
}

export interface SatelliteAbove {
  satid: number;
  satname: string;
  intDesignator: string;
  launchDate: string;
  satlat: number;
  satlng: number;
  satalt: number;
}

export interface AboveResponse {
  info: {
    category: string;
    transactionscount: number;
    satcount: number;
  };
  above: SatelliteAbove[];
}

export interface Observer {
  lat: number;
  lng: number;
  alt: number;
}

// -----------------------------------------------------------------------------
// Satellite Categories (for "What's Up?" function)
// -----------------------------------------------------------------------------
export const SATELLITE_CATEGORIES = {
  ALL: 0,
  BRIGHTEST: 1,
  ISS: 2,
  WEATHER: 3,
  NOAA: 4,
  GOES: 5,
  EARTH_RESOURCES: 6,
  SEARCH_RESCUE: 7,
  DISASTER_MONITORING: 8,
  TDRSS: 9,
  GEOSTATIONARY: 10,
  INTELSAT: 11,
  GORIZONT: 12,
  RADUGA: 13,
  MOLNIYA: 14,
  IRIDIUM: 15,
  ORBCOMM: 16,
  GLOBALSTAR: 17,
  AMATEUR_RADIO: 18,
  EXPERIMENTAL: 19,
  GPS_OPERATIONAL: 20,
  GLONASS_OPERATIONAL: 21,
  GALILEO: 22,
  SBAS: 23,
  NNSS: 24,
  RUSSIAN_LEO_NAV: 25,
  SPACE_EARTH_SCIENCE: 26,
  GEODETIC: 27,
  ENGINEERING: 28,
  EDUCATION: 29,
  MILITARY: 30,
  RADAR_CALIBRATION: 31,
  CUBESATS: 32,
  XM_SIRIUS: 33,
  TV: 34,
  BEIDOU: 35,
  YAOGAN: 36,
  WESTFORD_NEEDLES: 37,
  PARUS: 38,
  STRELA: 39,
  GONETS: 40,
  TSIKLON: 41,
  TSIKADA: 42,
  O3B: 43,
  TSELINA: 44,
  CELESTIS: 45,
  IRNSS: 46,
  QZSS: 47,
  FLOCK: 48,
  LEMUR: 49,
  GPS_CONSTELLATION: 50,
  GLONASS_CONSTELLATION: 51,
  STARLINK: 52,
  ONEWEB: 53,
  CHINESE_SPACE_STATION: 54,
  QIANFAN: 55,
  KUIPER: 56,
  GEESAT: 57,
} as const;

// -----------------------------------------------------------------------------
// API Configuration
// -----------------------------------------------------------------------------
const N2YO_API_BASE = 'https://api.n2yo.com/rest/v1/satellite';

// Well-known NORAD IDs
export const NORAD_IDS = {
  ISS: 25544,
  SMAP: 40376,
  HUBBLE: 20580,
  TIANGONG: 48274,
  STARLINK_FIRST: 44235,
} as const;

// Default observer (Paris, France)
const DEFAULT_OBSERVER: Observer = {
  lat: 48.8566,
  lng: 2.3522,
  alt: 0,
};

// -----------------------------------------------------------------------------
// API Functions
// -----------------------------------------------------------------------------

/**
 * Get Two-Line Elements (TLE) for a satellite
 * Transaction limit: 1000/hour
 */
export async function getTLE(
  apiKey: string,
  noradId: number
): Promise<TLEResponse | null> {
  try {
    const url = `${N2YO_API_BASE}/tle/${noradId}&apiKey=${apiKey}`;
    const response = await fetch(url);

    if (!response.ok) {
      console.error('[N2YO] TLE request failed:', response.status);
      return null;
    }

    const data = await response.json();
    if (data.error) {
      console.error('[N2YO] TLE error:', data.error);
      return null;
    }

    return data as TLEResponse;
  } catch (error) {
    console.error('[N2YO] TLE fetch error:', error);
    return null;
  }
}

/**
 * Get satellite positions (groundtrack)
 * Returns future positions as latitude, longitude, altitude
 * Transaction limit: 1000/hour
 *
 * @param seconds - Number of future positions (1-300)
 */
export async function getPositions(
  apiKey: string,
  noradId: number,
  observer: Observer = DEFAULT_OBSERVER,
  seconds: number = 1
): Promise<PositionsResponse | null> {
  try {
    // Clamp seconds to API limit
    const safeSeconds = Math.min(Math.max(1, seconds), 300);

    const url = `${N2YO_API_BASE}/positions/${noradId}/${observer.lat}/${observer.lng}/${observer.alt}/${safeSeconds}&apiKey=${apiKey}`;
    const response = await fetch(url);

    if (!response.ok) {
      console.error('[N2YO] Positions request failed:', response.status);
      return null;
    }

    const data = await response.json();
    if (data.error) {
      console.error('[N2YO] Positions error:', data.error);
      return null;
    }

    return data as PositionsResponse;
  } catch (error) {
    console.error('[N2YO] Positions fetch error:', error);
    return null;
  }
}

/**
 * Get visual passes for a satellite
 * Visual passes are optically visible to observers
 * Transaction limit: 100/hour
 *
 * @param days - Prediction days (1-10)
 * @param minVisibility - Minimum visibility duration in seconds
 */
export async function getVisualPasses(
  apiKey: string,
  noradId: number,
  observer: Observer = DEFAULT_OBSERVER,
  days: number = 10,
  minVisibility: number = 300
): Promise<VisualPassesResponse | null> {
  try {
    const safeDays = Math.min(Math.max(1, days), 10);

    const url = `${N2YO_API_BASE}/visualpasses/${noradId}/${observer.lat}/${observer.lng}/${observer.alt}/${safeDays}/${minVisibility}&apiKey=${apiKey}`;
    const response = await fetch(url);

    if (!response.ok) {
      console.error('[N2YO] Visual passes request failed:', response.status);
      return null;
    }

    const data = await response.json();
    if (data.error) {
      console.error('[N2YO] Visual passes error:', data.error);
      return null;
    }

    return data as VisualPassesResponse;
  } catch (error) {
    console.error('[N2YO] Visual passes fetch error:', error);
    return null;
  }
}

/**
 * Get radio passes for a satellite
 * Useful for radio communications prediction
 * Transaction limit: 100/hour
 *
 * @param days - Prediction days (1-10)
 * @param minElevation - Minimum elevation in degrees
 */
export async function getRadioPasses(
  apiKey: string,
  noradId: number,
  observer: Observer = DEFAULT_OBSERVER,
  days: number = 10,
  minElevation: number = 40
): Promise<RadioPassesResponse | null> {
  try {
    const safeDays = Math.min(Math.max(1, days), 10);

    const url = `${N2YO_API_BASE}/radiopasses/${noradId}/${observer.lat}/${observer.lng}/${observer.alt}/${safeDays}/${minElevation}&apiKey=${apiKey}`;
    const response = await fetch(url);

    if (!response.ok) {
      console.error('[N2YO] Radio passes request failed:', response.status);
      return null;
    }

    const data = await response.json();
    if (data.error) {
      console.error('[N2YO] Radio passes error:', data.error);
      return null;
    }

    return data as RadioPassesResponse;
  } catch (error) {
    console.error('[N2YO] Radio passes fetch error:', error);
    return null;
  }
}

/**
 * Get satellites above observer location ("What's Up?")
 * Returns all satellites within search radius
 * Transaction limit: 100/hour
 *
 * @param searchRadius - Search radius in degrees (0-90)
 * @param categoryId - Category filter (0 for all)
 */
export async function getSatellitesAbove(
  apiKey: string,
  observer: Observer = DEFAULT_OBSERVER,
  searchRadius: number = 70,
  categoryId: number = 0
): Promise<AboveResponse | null> {
  try {
    const safeRadius = Math.min(Math.max(0, searchRadius), 90);

    const url = `${N2YO_API_BASE}/above/${observer.lat}/${observer.lng}/${observer.alt}/${safeRadius}/${categoryId}&apiKey=${apiKey}`;
    const response = await fetch(url);

    if (!response.ok) {
      console.error('[N2YO] Above request failed:', response.status);
      return null;
    }

    const data = await response.json();
    if (data.error) {
      console.error('[N2YO] Above error:', data.error);
      return null;
    }

    return data as AboveResponse;
  } catch (error) {
    console.error('[N2YO] Above fetch error:', error);
    return null;
  }
}

// -----------------------------------------------------------------------------
// Legacy compatibility - getSatellitePosition (used by useSatelliteTracking)
// -----------------------------------------------------------------------------
export async function getSatellitePosition(
  apiKey: string,
  noradId: number = NORAD_IDS.SMAP,
  observer: Observer = DEFAULT_OBSERVER,
  seconds: number = 1
): Promise<PositionsResponse | null> {
  return getPositions(apiKey, noradId, observer, seconds);
}

// -----------------------------------------------------------------------------
// Coordinate Conversion Utilities
// -----------------------------------------------------------------------------

/**
 * Convert geographic coordinates to 3D Cartesian coordinates
 * For use with Three.js scene
 *
 * IMPORTANT: Uses significant visual exaggeration to make satellite orbit
 * clearly visible above Earth surface. Real LEO orbits (400-700km) are only
 * ~6-11% of Earth radius, which would be nearly invisible in 3D visualization.
 */
export function geoTo3D(
  lat: number,
  lon: number,
  altitude: number,
  earthRadius: number = 2.2
): { x: number; y: number; z: number } {
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  // Fixed orbit height above Earth surface (in scene units)
  // Must be high enough to never clip into the Earth globe
  const ORBIT_HEIGHT = 0.9; // ~40% of Earth radius - clearly visible orbit
  const radius = earthRadius + ORBIT_HEIGHT;

  // Spherical to Cartesian (Y-up for Three.js)
  const x = radius * Math.cos(latRad) * Math.sin(lonRad);
  const y = radius * Math.sin(latRad);
  const z = radius * Math.cos(latRad) * Math.cos(lonRad);

  return { x, y, z };
}

/**
 * Calculate orbital velocity from altitude
 */
export function calculateOrbitalVelocity(altitude: number): number {
  const G = 6.674e-11; // Gravitational constant
  const M = 5.972e24; // Earth mass in kg
  const r = (6371 + altitude) * 1000; // Orbital radius in meters
  const v = Math.sqrt((G * M) / r);
  return v / 1000; // km/s
}

/**
 * Calculate orbital period from altitude
 */
export function calculateOrbitalPeriod(altitude: number): number {
  const G = 6.674e-11;
  const M = 5.972e24;
  const r = (6371 + altitude) * 1000;
  const T = 2 * Math.PI * Math.sqrt(Math.pow(r, 3) / (G * M));
  return T / 60; // minutes
}

/**
 * Format position for display
 */
export function formatPosition(pos: SatellitePosition): {
  lat: string;
  lon: string;
  alt: string;
  velocity: string;
} {
  const velocity = calculateOrbitalVelocity(pos.sataltitude);

  return {
    lat: `${pos.satlatitude >= 0 ? 'N' : 'S'} ${Math.abs(pos.satlatitude).toFixed(4)}°`,
    lon: `${pos.satlongitude >= 0 ? 'E' : 'W'} ${Math.abs(pos.satlongitude).toFixed(4)}°`,
    alt: `${pos.sataltitude.toFixed(1)} km`,
    velocity: `${velocity.toFixed(2)} km/s`,
  };
}

/**
 * Parse TLE string into two lines
 */
export function parseTLE(tleString: string): { line1: string; line2: string } | null {
  const lines = tleString.split('\r\n');
  if (lines.length >= 2) {
    return { line1: lines[0], line2: lines[1] };
  }
  return null;
}

/**
 * Format Unix timestamp to local time
 */
export function formatTimestamp(unix: number, timezone?: string): string {
  const date = new Date(unix * 1000);
  return date.toLocaleString('fr-FR', {
    timeZone: timezone || Intl.DateTimeFormat().resolvedOptions().timeZone,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

// -----------------------------------------------------------------------------
// Simulated Position Generator (Fallback)
// -----------------------------------------------------------------------------
let simulatedAngle = 0;

export function getSimulatedPosition(): SatellitePosition {
  simulatedAngle += 0.01;

  // Simulate SMAP polar orbit (98.1° inclination)
  const inclination = 98.1;
  const altitude = 685;

  // Sun-synchronous orbit simulation
  const lat = Math.sin(simulatedAngle * 2) * (90 - Math.abs(90 - inclination));
  const lon = ((simulatedAngle * 180) / Math.PI) % 360 - 180;

  return {
    satlatitude: lat,
    satlongitude: lon,
    sataltitude: altitude,
    azimuth: (simulatedAngle * 57.3) % 360,
    elevation: Math.sin(simulatedAngle) * 45 + 45,
    ra: (simulatedAngle * 15) % 360,
    dec: lat * 0.4,
    timestamp: Math.floor(Date.now() / 1000),
  };
}

// -----------------------------------------------------------------------------
// Exports for backward compatibility
// -----------------------------------------------------------------------------
export const SMAP_NORAD_ID = NORAD_IDS.SMAP;
