// =============================================================================
// Root Layout with Navigation
// =============================================================================
import type { Metadata, Viewport } from 'next';
import './globals.css';
import { Navigation } from '@/components/layout/Navigation';

export const metadata: Metadata = {
  title: 'TINY-SAT Mission Control | Anomaly Detection Demo',
  description:
    'Educational demonstration of real-time satellite telemetry anomaly detection using LSTM neural networks. Built with Next.js, ONNX Runtime, and WebAssembly. Uses NASA SMAP public dataset.',
  keywords: [
    'satellite',
    'anomaly detection',
    'LSTM',
    'machine learning',
    'telemetry',
    'ONNX',
    'WebAssembly',
    'educational demo',
  ],
  authors: [{ name: 'MLOps Portfolio Project' }],
  openGraph: {
    title: 'TINY-SAT Mission Control - Educational Demo',
    description: 'Educational satellite anomaly detection demonstration in your browser',
    type: 'website',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#0d1117',
};

function DisclaimerBanner() {
  return (
    <div className="fixed bottom-0 left-0 right-0 z-30 bg-[#0d1117]/95 border-t border-amber-500/30 backdrop-blur-sm">
      <div className="max-w-[1920px] mx-auto px-4 py-1.5 flex items-center justify-center gap-2 text-[9px] text-amber-400/80">
        <span className="font-semibold">EDUCATIONAL DEMO</span>
        <span className="text-amber-400/50">|</span>
        <span className="text-[8px] text-amber-400/60">
          This project is not affiliated with NASA. Data sourced from NASA SMAP public dataset for educational purposes.
        </span>
      </div>
    </div>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body className="min-h-screen bg-space-900 text-space-100 font-mono antialiased">
        <Navigation />
        <main className="pt-16 pb-8">
          {children}
        </main>
        <DisclaimerBanner />
      </body>
    </html>
  );
}
