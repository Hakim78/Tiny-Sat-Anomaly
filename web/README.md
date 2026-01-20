# TINY-SAT Mission Control - Web Dashboard

> Production-grade satellite anomaly detection running **entirely in the browser** using ONNX WebAssembly.

![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
![ONNX](https://img.shields.io/badge/ONNX-Runtime%20Web-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue?logo=typescript)
![Tailwind](https://img.shields.io/badge/Tailwind-3.4-38bdf8?logo=tailwindcss)

## Features

- **Client-Side AI Inference**: LSTM model runs in WebAssembly (no server required)
- **3D Globe Visualization**: Real-time satellite tracking with react-globe.gl
- **Live Telemetry Charts**: Recharts-powered oscilloscope display
- **Chaos Engineering**: Solar Flare injection for testing anomaly detection
- **Cyberpunk HUD Theme**: Dark mode space command center aesthetic

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  React UI   │  │   Zustand   │  │  ONNX Runtime WASM  │  │
│  │  Components │◄─│    Store    │◄─│   (LSTM Model)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                                      ▲             │
│         ▼                                      │             │
│  ┌─────────────────────────────────────────────┴───────────┐│
│  │              /public/model.onnx + telemetry_data.json   ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prepare Assets (Python)

First, convert the PyTorch model and data:

```bash
# From the project root
cd web
pip install onnx  # If not installed
python prepare_assets.py
```

This creates:
- `public/model.onnx` - ONNX model (~500KB)
- `public/telemetry_data.json` - Telemetry data (~200KB)

### 2. Install Dependencies

```bash
npm install
```

### 3. Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### 4. Production Build

```bash
npm run build
npm start
```

## Deployment to Vercel

1. Push to GitHub
2. Import project in Vercel
3. Deploy (automatic)

The app is fully static and runs entirely in the browser.

## Project Structure

```
web/
├── public/
│   ├── model.onnx           # ONNX model (generated)
│   └── telemetry_data.json  # Telemetry data (generated)
├── src/
│   ├── app/
│   │   ├── globals.css      # Tailwind + custom styles
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Main dashboard page
│   ├── components/
│   │   ├── globe/           # 3D Earth visualization
│   │   ├── dashboard/       # Charts & displays
│   │   └── layout/          # HUD, controls, status
│   ├── hooks/
│   │   └── useInference.ts  # ONNX inference hook
│   ├── lib/
│   │   └── onnx.ts          # ONNX session management
│   └── store/
│       └── useStore.ts      # Zustand global state
├── prepare_assets.py        # PyTorch → ONNX converter
├── package.json
├── tailwind.config.ts
└── next.config.js
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript 5.3 |
| Styling | Tailwind CSS 3.4 |
| State | Zustand 4.5 |
| AI Runtime | ONNX Runtime Web (WASM) |
| 3D Globe | react-globe.gl + Three.js |
| Charts | Recharts 2.12 |

## Browser Compatibility

- Chrome 90+ ✅
- Firefox 89+ ✅
- Safari 15+ ✅
- Edge 90+ ✅

Requires WebAssembly support.

## Performance

- Model size: ~500KB (gzipped: ~150KB)
- Inference time: ~5-10ms per prediction
- Memory: ~50MB (model + data)

## License

MIT
