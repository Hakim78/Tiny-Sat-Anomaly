// =============================================================================
// Metric Card - NASA/SpaceX Style KPI Display
// =============================================================================
'use client';

import { ReactNode } from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  suffix?: string;
  icon?: ReactNode;
  variant?: 'default' | 'success' | 'danger' | 'warning';
  delta?: string;
  deltaType?: 'positive' | 'negative' | 'neutral';
  className?: string;
}

export function MetricCard({
  label,
  value,
  suffix = '',
  icon,
  variant = 'default',
  delta,
  deltaType = 'neutral',
  className = '',
}: MetricCardProps) {
  const variantStyles = {
    default: 'border-slate-800',
    success: 'border-green-500/30 bg-green-500/5',
    danger: 'border-red-500/30 bg-red-500/5',
    warning: 'border-yellow-500/30 bg-yellow-500/5',
  };

  const valueColors = {
    default: 'text-white',
    success: 'text-green-400',
    danger: 'text-red-400',
    warning: 'text-yellow-400',
  };

  const deltaColors = {
    positive: 'text-green-400',
    negative: 'text-red-400',
    neutral: 'text-slate-500',
  };

  return (
    <div
      className={`rounded-lg bg-slate-900/50 border p-3 ${variantStyles[variant]} ${className}`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-wider">
          {icon && <span className="text-slate-400">{icon}</span>}
          {label}
        </div>
        {variant === 'danger' && (
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
        )}
      </div>

      <div className="mt-2">
        <span className={`text-xl font-bold font-mono ${valueColors[variant]}`}>
          {value}
        </span>
        {suffix && (
          <span className="text-sm text-slate-500 ml-1">{suffix}</span>
        )}
      </div>

      {delta && (
        <div className={`text-[10px] mt-1 font-mono ${deltaColors[deltaType]}`}>
          {delta}
        </div>
      )}
    </div>
  );
}

export default MetricCard;
