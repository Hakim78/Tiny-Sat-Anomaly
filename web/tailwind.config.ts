import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Space HUD Color Palette
        space: {
          900: '#080b0f',
          800: '#0d1117',
          700: '#161b22',
          600: '#21262d',
          500: '#30363d',
          400: '#484f58',
          300: '#6e7681',
          200: '#8b949e',
          100: '#c9d1d9',
        },
        neon: {
          blue: '#58a6ff',
          green: '#3fb950',
          red: '#f85149',
          yellow: '#d29922',
          cyan: '#39c5cf',
          purple: '#a371f7',
        },
      },
      fontFamily: {
        mono: ['SF Mono', 'Consolas', 'Liberation Mono', 'Menlo', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'glow-green': 'glowGreen 2s ease-in-out infinite alternate',
        'glow-red': 'glowRed 0.5s ease-in-out infinite alternate',
        'orbit': 'orbit 60s linear infinite',
        'scanline': 'scanline 8s linear infinite',
      },
      keyframes: {
        glowGreen: {
          '0%': { boxShadow: '0 0 20px rgba(63, 185, 80, 0.3)' },
          '100%': { boxShadow: '0 0 40px rgba(63, 185, 80, 0.6)' },
        },
        glowRed: {
          '0%': { boxShadow: '0 0 20px rgba(248, 81, 73, 0.4)' },
          '100%': { boxShadow: '0 0 60px rgba(248, 81, 73, 0.8)' },
        },
        orbit: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};

export default config;
