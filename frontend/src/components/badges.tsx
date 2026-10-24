import type { ModelSummary } from '../api'

export const RISK_BADGE: Record<string, string> = {
  Low: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20',
  Moderate: 'bg-amber-50 text-amber-800 ring-amber-600/20',
  High: 'bg-rose-50 text-rose-700 ring-rose-600/20',
}

export function StatusBadge({ status }: { status: ModelSummary['status'] }) {
  const ready = status === 'ready'
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset ${
      ready ? 'bg-emerald-50 text-emerald-700 ring-emerald-600/20' : 'bg-slate-100 text-slate-600 ring-slate-500/20'
    }`}>
      <span className={`h-1.5 w-1.5 rounded-full ${ready ? 'bg-emerald-500' : 'bg-slate-400'}`} />
      {ready ? 'Ready' : 'Not trained'}
    </span>
  )
}

export function CategoryBadge({ category }: { category: string }) {
  return <span className="rounded-md bg-teal-50 px-2 py-0.5 text-xs font-semibold text-teal-700">{category}</span>
}

export const pct = (v: number | null | undefined) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '–')
