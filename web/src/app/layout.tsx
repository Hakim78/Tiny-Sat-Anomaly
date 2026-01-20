// =============================================================================
// Root Layout with Navigation
// =============================================================================
import type { Metadata, Viewport } from 'next';
import './globals.css';
import { Navigation } from '@/components/layout/Navigation';

export const metadata: Metadata = {
  title: 'TINY-SAT Mission Control | NASA Anomaly Detection',
  description:
    'Real-time satellite telemetry anomaly detection using LSTM neural networks. Built with Next.js, ONNX Runtime, and WebAssembly.',
  keywords: [
    'NASA',
    'satellite',
    'anomaly detection',
    'LSTM',
    'machine learning',
    'telemetry',
    'ONNX',
    'WebAssembly',
  ],
  authors: [{ name: 'MLOps Team' }],
  openGraph: {
    title: 'TINY-SAT Mission Control',
    description: 'Real-time satellite anomaly detection in your browser',
    type: 'website',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#0d1117',
};

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
        <main className="pt-16">
          {children}
        </main>
      </body>
    </html>
  );
}
