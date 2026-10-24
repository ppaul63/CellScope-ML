import { useCallback, useEffect, useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getTrainingRun, listTrainingRuns, type EpochMetrics, type RunStatus, type TrainingRun } from '../api'
import { pct } from '../components/badges'
import { Icon, type IconName } from '../components/icons'
import { href } from '../router'

const POLL_MS = 3000
// Validated categorical pair (blue, orange); validation series are also dashed so identity isn't colour-only
const SERIES = { train: '#2a78d6', val: '#eb6834' }

const STATUS: Record<RunStatus, { label: string; icon: IconName; style: string }> = {
  running: { label: 'Running', icon: 'clock', style: 'bg-sky-50 text-sky-700 ring-sky-600/20' },
  completed: { label: 'Completed', icon: 'check', style: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20' },
  early_stopped: { label: 'Early stopped', icon: 'check', style: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20' },
  stopped: { label: 'Stopped', icon: 'stop', style: 'bg-slate-100 text-slate-700 ring-slate-500/20' },
  interrupted: { label: 'Interrupted', icon: 'alert', style: 'bg-amber-50 text-amber-800 ring-amber-600/20' },
  failed: { label: 'Failed', icon: 'alert', style: 'bg-rose-50 text-rose-700 ring-rose-600/20' },
}

const KIND: Record<string, string> = {
  scratch: 'Scratch CNN',
  pretrained: 'Pretrained baseline',
  scratch_overfit: 'Overfit check · scratch',
  pretrained_overfit: 'Overfit check · pretrained',
}

function duration(s: number | null | undefined) {
  if (s == null) return '–'
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.round(s % 60)
  if (h) return `${h}h ${m}m`
  if (m) return `${m}m ${sec}s`
  return `${sec}s`
}

function when(ts: string) {
  return new Date(ts).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

// Poll while something is running, and pause while the browser tab is hidden
function usePolling(active: boolean, tick: () => void) {
  useEffect(() => {
    if (!active) return
    const id = window.setInterval(() => {
      if (!document.hidden) tick()
    }, POLL_MS)
    return () => window.clearInterval(id)
  }, [active, tick])
}

function StatusBadge({ status }: { status: RunStatus }) {
  const s = STATUS[status] ?? STATUS.stopped
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset ${s.style}`}>
      {status === 'running' ? (
        <span className="relative mr-0.5 flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-sky-500" />
        </span>
      ) : (
        <Icon name={s.icon} className="h-3.5 w-3.5" />
      )}
      {s.label}
    </span>
  )
}

function ProgressBar({ run, thick = false }: { run: TrainingRun; thick?: boolean }) {
  const running = run.status === 'running'
  const bad = run.status === 'failed' || run.status === 'interrupted'
  return (
    <div
      className={`w-full overflow-hidden rounded-full bg-slate-100 ${thick ? 'h-3' : 'h-2'}`}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(run.progress * 100)}
      aria-label={`Training progress for ${run.model_name}`}
    >
      <div
        className={`h-full rounded-full transition-all duration-700 ${
          bad ? 'bg-amber-500' : running ? 'bg-sky-500' : run.status === 'stopped' ? 'bg-slate-400' : 'bg-emerald-500'
        }`}
        style={{ width: `${Math.max(run.progress * 100, run.progress > 0 ? 2 : 0)}%` }}
      />
    </div>
  )
}

function epochLabel(run: TrainingRun) {
  if (run.status === 'running') {
    const cur = Math.min(run.epoch + 1, run.epochs_total)
    return `Epoch ${cur} of ${run.epochs_total} · batch ${run.batch}/${run.batches_per_epoch}`
  }
  return `${run.epoch} of ${run.epochs_total} epochs`
}

// ---- list ----------------------------------------------------------------

function EmptyState() {
  return (
    <div className="rounded-2xl bg-white p-8 text-center shadow-sm ring-1 ring-slate-200">
      <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-teal-50 text-teal-700">
        <Icon name="chart" className="h-6 w-6" />
      </span>
      <h2 className="mt-4 text-lg font-semibold text-slate-900">No training runs yet</h2>
      <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
        Start training from a terminal in the project folder. Runs appear here as soon as they begin, and update live.
      </p>
      <pre className="mx-auto mt-4 max-w-md overflow-x-auto rounded-lg bg-slate-900 px-4 py-3 text-left text-xs text-slate-100">
{`python -m ml.prepare_data
python -m ml.train --model pretrained
python -m ml.train --model scratch`}
      </pre>
    </div>
  )
}

function RunRow({ run }: { run: TrainingRun }) {
  return (
    <a
      href={href({ page: 'training', modelId: run.model_id, runId: run.run_id })}
      className="block px-5 py-4 transition hover:bg-slate-50"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold text-slate-900">{run.model_name}</span>
            <span className="text-sm text-slate-500">{KIND[run.kind] ?? run.kind}</span>
            <StatusBadge status={run.status} />
          </div>
          <div className="mt-0.5 text-xs text-slate-500">Started {when(run.started_at)} · {run.arch} on {run.device}</div>
        </div>
        <div className="text-right text-sm">
          <div className="font-semibold tabular-nums text-slate-900">{pct(run.best?.val_macro_f1)}</div>
          <div className="text-xs text-slate-500">best val macro F1</div>
        </div>
      </div>
      <div className="mt-3"><ProgressBar run={run} /></div>
      <div className="mt-1.5 flex flex-wrap justify-between gap-2 text-xs text-slate-500">
        <span className="tabular-nums">{epochLabel(run)}</span>
        <span className="tabular-nums">
          {duration(run.elapsed_seconds)} elapsed{run.eta_seconds != null && ` · about ${duration(run.eta_seconds)} left`}
        </span>
      </div>
    </a>
  )
}

function RunList() {
  const [runs, setRuns] = useState<TrainingRun[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    listTrainingRuns().then((r) => { setRuns(r); setError(null) }).catch((e) => setError(e.message))
  }, [])
  useEffect(load, [load])
  usePolling(!!runs?.some((r) => r.status === 'running'), load)

  const running = runs?.filter((r) => r.status === 'running').length ?? 0

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Training</h1>
          <p className="mt-1 text-sm text-slate-500">Progress and results of every model training run.</p>
        </div>
        {running > 0 && (
          <span className="inline-flex items-center gap-2 text-sm text-slate-600">
            <span className="h-2 w-2 animate-pulse rounded-full bg-sky-500" />
            {running} running · updates every {POLL_MS / 1000}s
          </span>
        )}
      </div>

      {error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</div>}

      {runs === null && !error ? (
        <div className="rounded-2xl bg-white p-10 text-center text-sm text-slate-500 ring-1 ring-slate-200">Loading runs…</div>
      ) : runs && runs.length === 0 ? (
        <EmptyState />
      ) : (
        <section className="overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
          <div className="border-b border-slate-200 px-5 py-3 text-sm font-semibold text-slate-900">
            Runs <span className="font-normal text-slate-500">({runs?.length ?? 0})</span>
          </div>
          <div className="divide-y divide-slate-200">{runs?.map((r) => <RunRow key={`${r.model_id}/${r.run_id}`} run={r} />)}</div>
        </section>
      )}
    </div>
  )
}

// ---- detail --------------------------------------------------------------

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="min-w-0 rounded-2xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
      <div className="text-sm text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-bold tabular-nums text-slate-900">{value}</div>
      {note && <div className="mt-0.5 truncate text-xs text-slate-500">{note}</div>}
    </div>
  )
}

function MetricChart({ title, data, lines, percent }: {
  title: string
  data: EpochMetrics[]
  lines: { key: keyof EpochMetrics; name: string; color: string; dashed?: boolean }[]
  percent?: boolean
}) {
  const fmt = (v: number) => (percent ? `${(v * 100).toFixed(1)}%` : v.toFixed(3))
  return (
    <section className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
      <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      <div className="mt-3 h-64">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-500">Waiting for the first epoch to finish…</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="#eef0f3" vertical={false} />
              <XAxis dataKey="epoch" tick={{ fontSize: 12, fill: '#64748b' }} tickLine={false} axisLine={{ stroke: '#e2e8f0' }}
                label={{ value: 'Epoch', position: 'insideBottomRight', offset: -2, fontSize: 11, fill: '#94a3b8' }} />
              <YAxis tick={{ fontSize: 12, fill: '#64748b' }} tickLine={false} axisLine={false} width={48}
                domain={percent ? [0, 1] : ['auto', 'auto']} tickFormatter={(v: number) => (percent ? `${Math.round(v * 100)}%` : v.toFixed(2))} />
              <Tooltip
                formatter={(v) => fmt(Number(v))}
                labelFormatter={(l) => `Epoch ${l}`}
                contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }}
                itemStyle={{ color: '#334155' }}
              />
              <Legend iconType="plainline" wrapperStyle={{ fontSize: 12, color: '#475569' }} />
              {lines.map((l) => (
                <Line key={l.key} type="monotone" dataKey={l.key} name={l.name} stroke={l.color} strokeWidth={2}
                  strokeDasharray={l.dashed ? '6 4' : undefined} dot={data.length <= 12 ? { r: 3, strokeWidth: 0, fill: l.color } : false}
                  activeDot={{ r: 5, stroke: '#fff', strokeWidth: 2 }} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  )
}

function RunDetail({ modelId, runId }: { modelId: string; runId: string }) {
  const [run, setRun] = useState<TrainingRun | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    getTrainingRun(modelId, runId).then((r) => { setRun(r); setError(null) }).catch((e) => setError(e.message))
  }, [modelId, runId])
  useEffect(load, [load])
  usePolling(run?.status === 'running', load)

  if (error) return <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</div>
  if (!run) return <div className="rounded-2xl bg-white p-10 text-center text-sm text-slate-500 ring-1 ring-slate-200">Loading run…</div>

  const history = run.history ?? []
  const latest = history[history.length - 1]
  const running = run.status === 'running'
  const details: [string, string | number][] = [
    ['Run ID', run.run_id], ['Architecture', run.arch], ['Device', run.device],
    ['Batch size', run.batch_size], ['Learning rate', run.lr], ['Training images', run.train_size],
    ['Validation images', run.val_size], ['Early-stop patience', run.patience ?? '–'], ['Checkpoint', run.checkpoint],
  ]

  return (
    <div className="space-y-6">
      <div>
        <nav className="text-sm text-slate-500" aria-label="Breadcrumb">
          <a href={href({ page: 'training' })} className="hover:text-slate-700">Training</a>
          <span className="mx-2">/</span>
          <span className="text-slate-700">{run.run_id}</span>
        </nav>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-bold text-slate-900">{run.model_name}</h1>
          <span className="text-slate-500">{KIND[run.kind] ?? run.kind}</span>
          <StatusBadge status={run.status} />
        </div>
        <p className="mt-1 text-sm text-slate-500">
          Started {when(run.started_at)}{run.finished_at && ` · finished ${when(run.finished_at)}`}
          {running && ` · updates every ${POLL_MS / 1000}s`}
        </p>
      </div>

      {run.message && (
        <div className={`rounded-xl border px-4 py-3 text-sm ${
          run.status === 'failed' ? 'border-rose-200 bg-rose-50 text-rose-800'
            : run.status === 'interrupted' ? 'border-amber-200 bg-amber-50 text-amber-900'
              : 'border-slate-200 bg-white text-slate-700'
        }`}>
          {run.message}
        </div>
      )}

      <section className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
        <div className="flex flex-wrap items-baseline justify-between gap-2 text-sm">
          <span className="font-semibold text-slate-900">{epochLabel(run)}</span>
          <span className="font-semibold tabular-nums text-slate-900">{Math.round(run.progress * 100)}%</span>
        </div>
        <div className="mt-2"><ProgressBar run={run} thick /></div>
      </section>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Tile label="Best val macro F1" value={pct(run.best?.val_macro_f1)} note={run.best ? `at epoch ${run.best.epoch}` : 'after the first epoch'} />
        <Tile label="Latest val accuracy" value={pct(latest?.val_acc)} note={latest ? `epoch ${latest.epoch}` : undefined} />
        <Tile label="Elapsed" value={duration(run.elapsed_seconds)} note={run.avg_epoch_seconds ? `${duration(run.avg_epoch_seconds)} per epoch` : undefined} />
        <Tile label={running ? 'Time left' : 'Epochs run'} value={running ? duration(run.eta_seconds) : String(run.epoch)}
          note={running ? 'estimate, at the current pace' : `of ${run.epochs_total} planned`} />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <MetricChart title="Loss" data={history} lines={[
          { key: 'train_loss', name: 'Training loss', color: SERIES.train },
          { key: 'val_loss', name: 'Validation loss', color: SERIES.val, dashed: true },
        ]} />
        <MetricChart title="Validation accuracy and macro F1" percent data={history} lines={[
          { key: 'val_acc', name: 'Accuracy', color: SERIES.train },
          { key: 'val_macro_f1', name: 'Macro F1', color: SERIES.val, dashed: true },
        ]} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <section className="overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 lg:col-span-2">
          <h3 className="border-b border-slate-200 px-5 py-3 text-sm font-semibold text-slate-900">Epochs</h3>
          <div className="max-h-96 overflow-auto">
            <table className="w-full text-right text-sm tabular-nums">
              <thead className="sticky top-0 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  {['Epoch', 'Train loss', 'Val loss', 'Val acc', 'Macro F1', 'Time'].map((h) => (
                    <th key={h} scope="col" className="px-4 py-2 font-semibold first:text-left">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {[...history].reverse().map((h) => (
                  <tr key={h.epoch} className={run.best?.epoch === h.epoch ? 'bg-emerald-50/60' : undefined}>
                    <td className="px-4 py-2 text-left font-medium text-slate-900">
                      {h.epoch}{run.best?.epoch === h.epoch && <span className="ml-2 text-xs font-semibold text-emerald-700">best</span>}
                    </td>
                    <td className="px-4 py-2">{h.train_loss.toFixed(4)}</td>
                    <td className="px-4 py-2">{h.val_loss.toFixed(4)}</td>
                    <td className="px-4 py-2">{pct(h.val_acc)}</td>
                    <td className="px-4 py-2">{pct(h.val_macro_f1)}</td>
                    <td className="px-4 py-2 text-slate-500">{duration(h.seconds)}</td>
                  </tr>
                ))}
                {history.length === 0 && (
                  <tr><td colSpan={6} className="px-4 py-6 text-center text-slate-500">No epochs finished yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
          <h3 className="text-sm font-semibold text-slate-900">Run details</h3>
          <dl className="mt-3 space-y-2 text-sm">
            {details.map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4">
                <dt className="text-slate-500">{k}</dt>
                <dd className="truncate text-right font-medium text-slate-800">{v}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>
    </div>
  )
}

export default function TrainingPage({ modelId, runId }: { modelId?: string; runId?: string }) {
  return modelId && runId ? <RunDetail key={`${modelId}/${runId}`} modelId={modelId} runId={runId} /> : <RunList />
}
