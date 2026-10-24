import { useState } from 'react'
import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import type { Prediction, RiskLevel } from '../api'

const RISK_STYLE: Record<RiskLevel, { badge: string; bar: string }> = {
  Low: { badge: 'bg-emerald-100 text-emerald-800 ring-emerald-600/20', bar: '#059669' },
  Moderate: { badge: 'bg-amber-100 text-amber-800 ring-amber-600/20', bar: '#d97706' },
  High: { badge: 'bg-rose-100 text-rose-800 ring-rose-600/20', bar: '#e11d48' },
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

interface Props {
  result: Prediction
  previewUrl: string
}

export default function ResultCard({ result, previewUrl }: Props) {
  const [showHeatmap, setShowHeatmap] = useState(false)
  const style = RISK_STYLE[result.risk_level] ?? RISK_STYLE.Moderate
  const data = Object.entries(result.probabilities)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
  const heatmapSrc = result.heatmap_png_base64 ? `data:image/png;base64,${result.heatmap_png_base64}` : null

  return (
    <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200" aria-live="polite">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Predicted category</p>
          <h2 className="mt-1 text-3xl font-bold text-slate-900">{result.predicted_class}</h2>
          <p className="text-sm text-slate-600">{result.class_name}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ring-1 ring-inset ${style.badge}`}>
          {result.risk_level} risk
        </span>
      </div>

      <div className="mt-5">
        <div className="flex items-baseline justify-between text-sm">
          <span className="font-medium text-slate-700">Confidence</span>
          <span className="font-semibold tabular-nums text-slate-900">{pct(result.confidence)}</span>
        </div>
        <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
          <div className="h-full rounded-full transition-all" style={{ width: pct(result.confidence), background: style.bar }} />
        </div>
        {result.confidence < 0.6 && (
          <p className="mt-2 text-xs text-amber-700">Low confidence — this image is ambiguous for the model; manual review is especially important.</p>
        )}
      </div>

      <p className="mt-4 text-sm text-slate-700">{result.description}</p>

      <h3 className="mt-6 text-sm font-semibold text-slate-800">Probability by class</h3>
      <div className="mt-2 h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 4, right: 48, top: 4, bottom: 4 }}>
            <XAxis type="number" domain={[0, 1]} hide />
            <YAxis type="category" dataKey="name" width={56} tickLine={false} axisLine={false} tick={{ fontSize: 13, fill: '#334155' }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.name} fill={d.name === result.predicted_class ? style.bar : '#cbd5e1'} />
              ))}
              <LabelList dataKey="value" position="right" formatter={(v) => pct(Number(v))} style={{ fontSize: 12, fill: '#475569' }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {heatmapSrc && (
        <div className="mt-6">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-800">Where the model looked</h3>
            <div className="inline-flex rounded-lg bg-slate-100 p-0.5 text-sm" role="group">
              {(['Original', 'Heatmap'] as const).map((label) => {
                const active = (label === 'Heatmap') === showHeatmap
                return (
                  <button
                    key={label}
                    type="button"
                    onClick={() => setShowHeatmap(label === 'Heatmap')}
                    className={`rounded-md px-3 py-1 font-medium transition ${active ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                  >
                    {label}
                  </button>
                )
              })}
            </div>
          </div>
          <img
            src={showHeatmap ? heatmapSrc : previewUrl}
            alt={showHeatmap ? 'Grad-CAM heatmap' : 'Uploaded image'}
            className="mt-3 max-h-80 w-full rounded-lg bg-slate-50 object-contain"
          />
          <p className="mt-2 text-xs text-slate-500">
            Grad-CAM: red regions influenced the prediction most. The heatmap shows the centre crop the model analysed.
          </p>
        </div>
      )}

      <p className="mt-5 text-right text-xs text-slate-400">Analysed in {result.inference_ms.toFixed(0)} ms</p>
    </section>
  )
}
