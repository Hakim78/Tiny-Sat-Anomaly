// =============================================================================
// Navigation Component - Professional Mission Control Navigation
// =============================================================================
'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import {
  LayoutDashboard,
  LineChart,
  Brain,
  Shield,
  ScrollText,
  Satellite,
  Activity,
} from 'lucide-react';
import { useStore } from '@/store/useStore';

interface NavItem {
  href: string;
  label: string;
  shortLabel: string;
  icon: React.ReactNode;
  description: string;
}

const navItems: NavItem[] = [
  {
    href: '/',
    label: 'Dashboard',
    shortLabel: 'DASH',
    icon: <LayoutDashboard className="w-4 h-4" />,
    description: 'Mission Control Overview',
  },
  {
    href: '/analytics',
    label: 'Analytics',
    shortLabel: 'ANLY',
    icon: <LineChart className="w-4 h-4" />,
    description: 'Temporal Analysis & Trends',
  },
  {
    href: '/model',
    label: 'ML Model',
    shortLabel: 'MODEL',
    icon: <Brain className="w-4 h-4" />,
    description: 'Model Card & Performance',
  },
  {
    href: '/operations',
    label: 'Operations',
    shortLabel: 'OPS',
    icon: <Shield className="w-4 h-4" />,
    description: 'SOP & Operator Context',
  },
  {
    href: '/logs',
    label: 'Event Logs',
    shortLabel: 'LOGS',
    icon: <ScrollText className="w-4 h-4" />,
    description: 'Mission Event History',
  },
];

export function Navigation() {
  const pathname = usePathname();
  const isAnomaly = useStore((s) => s.isAnomaly);
  const anomalyCount = useStore((s) => s.anomalyCount);
  const isPlaying = useStore((s) => s.isPlaying);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/10 bg-[var(--void-black)]/95 backdrop-blur-md">
      <div className="max-w-[1920px] mx-auto">
        {/* Top Bar */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-white/5">
          {/* Logo & Title */}
          <div className="flex items-center gap-3">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              isAnomaly
                ? 'bg-[var(--alert-red)]/20 border border-[var(--alert-red)]/50'
                : 'bg-[var(--accent-cyan)]/20 border border-[var(--accent-cyan)]/50'
            }`}>
              <Satellite className={`w-4 h-4 ${isAnomaly ? 'text-[var(--alert-red)]' : 'text-[var(--accent-cyan)]'}`} />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-wider">
                <span className="text-[var(--signal-white)]">TINY-SAT</span>
                <span className="text-[var(--accent-cyan)] ml-2">COMMAND</span>
              </h1>
              <p className="text-[8px] text-[var(--signal-dim)] uppercase tracking-widest">
                Anomaly Detection System v2.0
              </p>
            </div>
          </div>

          {/* Status Indicators - Hidden on mobile */}
          <div className="hidden md:flex items-center gap-4">
            {/* System Status */}
            <div className="flex items-center gap-2 px-3 py-1 rounded bg-black/30 border border-white/5">
              <Activity className={`w-3 h-3 ${isPlaying ? 'text-[var(--nominal-green)] animate-pulse' : 'text-[var(--signal-dim)]'}`} />
              <span className="text-[9px] text-[var(--signal-dim)] uppercase">System</span>
              <span className={`text-[9px] font-bold ${isPlaying ? 'text-[var(--nominal-green)]' : 'text-[var(--signal-dim)]'}`}>
                {isPlaying ? 'ACTIVE' : 'STANDBY'}
              </span>
            </div>

            {/* Anomaly Counter */}
            <div className={`flex items-center gap-2 px-3 py-1 rounded border ${
              anomalyCount > 0
                ? 'bg-[var(--alert-red)]/10 border-[var(--alert-red)]/30'
                : 'bg-black/30 border-white/5'
            }`}>
              <span className="text-[9px] text-[var(--signal-dim)] uppercase">Anomalies</span>
              <span className={`text-[9px] font-bold ${anomalyCount > 0 ? 'text-[var(--alert-red)]' : 'text-[var(--signal-dim)]'}`}>
                {anomalyCount}
              </span>
            </div>

            {/* Operator ID */}
            <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded bg-black/30 border border-white/5">
              <div className="w-2 h-2 rounded-full bg-[var(--nominal-green)]" />
              <span className="text-[9px] text-[var(--signal-dim)]">Operator:</span>
              <span className="text-[9px] text-[var(--signal-white)] font-mono">MCO-L2</span>
            </div>

            {/* UTC Time */}
            <div className="hidden lg:block text-[10px] font-mono text-[var(--accent-cyan)]">
              {new Date().toISOString().replace('T', ' ').slice(0, 19)} UTC
            </div>
          </div>

          {/* Mobile: Anomaly badge only */}
          <div className="flex md:hidden items-center gap-2">
            <div className={`flex items-center gap-1 px-2 py-1 rounded border text-[8px] ${
              anomalyCount > 0
                ? 'bg-[var(--alert-red)]/10 border-[var(--alert-red)]/30 text-[var(--alert-red)]'
                : 'bg-black/30 border-white/5 text-[var(--signal-dim)]'
            }`}>
              <span className="font-bold">{anomalyCount}</span>
              <span>ANM</span>
            </div>
          </div>
        </div>

        {/* Navigation Links - Scrollable on mobile */}
        <div className="flex items-center gap-1 px-2 md:px-4 py-1 overflow-x-auto scrollbar-hide">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`group relative flex items-center gap-1 md:gap-2 px-2 md:px-4 py-2 rounded transition-all flex-shrink-0 ${
                  isActive
                    ? 'bg-[var(--accent-cyan)]/10 border border-[var(--accent-cyan)]/30 text-[var(--accent-cyan)]'
                    : 'text-[var(--signal-dim)] hover:text-[var(--signal-white)] hover:bg-white/5'
                }`}
              >
                {item.icon}
                <span className="text-[9px] md:text-[10px] font-semibold uppercase tracking-wider">
                  {item.shortLabel}
                </span>

                {/* Tooltip - Hidden on mobile */}
                <div className="hidden md:block absolute top-full left-1/2 -translate-x-1/2 mt-2 px-3 py-2 bg-[var(--panel-dark)] border border-white/10 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
                  <div className="text-[10px] font-semibold text-[var(--signal-white)]">{item.label}</div>
                  <div className="text-[8px] text-[var(--signal-dim)]">{item.description}</div>
                </div>

                {/* Active indicator */}
                {isActive && (
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-6 md:w-8 h-0.5 bg-[var(--accent-cyan)]" />
                )}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
