import type { ModelMetrics, ModelSummary } from '../api'
import { RISK_BADGE, pct } from './badges'

const METRIC_LABELS: [keyof ModelMetrics, string][] = [
  ['accuracy', 'Accuracy'],
  ['macro_f1', 'Macro F1'],
  ['roc_auc_ovr', 'ROC-AUC'],
  ['abnormal_sensitivity', 'Abnormal sensitivity'],
  ['high_grade_recall', 'High-grade recall'],
]

export default function AboutModel({ model }: { model: ModelSummary }) {
  const metrics = METRIC_LABELS.filter(([key]) => typeof model.test?.[key] === 'number')

  return (
    <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
      <h2 className="text-lg font-semibold text-slate-900">About this model</h2>
      {model.status === 'ready' ? (
        <p className="mt-1 text-sm text-slate-600">
          {model.model_kind === 'scratch' ? 'Custom residual CNN trained from scratch' : 'Fine-tuned ImageNet backbone'} ({model.arch}),{' '}
          {model.image_size}×{model.image_size} input.
          {model.test?.test_size ? ` Evaluated on ${model.test.test_size} held-out images.` : ''}
        </p>
      ) : (
        <p className="mt-1 text-sm text-slate-500">Not trained yet, so there are no performance metrics.</p>
      )}

      {metrics.length > 0 && (
        <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {metrics.map(([key, label]) => (
            <div key={key} className="rounded-xl bg-slate-50 p-3">
              <dt className="text-xs text-slate-500">{label}</dt>
              <dd className="mt-1 text-xl font-semibold tabular-nums text-slate-900">{pct(model.test?.[key])}</dd>
            </div>
          ))}
        </dl>
      )}

      <h3 className="mt-6 text-sm font-semibold text-slate-800">What it detects</h3>
      <ul className="mt-2 divide-y divide-slate-100">
        {model.classes.map((c) => (
          <li key={c.code} className="flex flex-col gap-1 py-2.5 sm:flex-row sm:items-start sm:gap-4">
            <span className="w-16 shrink-0 font-semibold text-slate-900">{c.code}</span>
            <div className="min-w-0 flex-1">
              <div className="text-sm font-medium text-slate-800">{c.name}</div>
              <div className="text-sm text-slate-500">{c.description}</div>
            </div>
            <span className={`self-start rounded-full px-2 py-0.5 text-xs font-semibold ring-1 ring-inset ${RISK_BADGE[c.risk] ?? 'bg-slate-100 text-slate-600 ring-slate-500/20'}`}>
              {c.risk} risk
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}
