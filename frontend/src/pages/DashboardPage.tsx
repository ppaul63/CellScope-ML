import { useEffect, useState } from 'react'
import { listUsers, type ModelSummary, type User } from '../api'
import { CategoryBadge, StatusBadge, pct } from '../components/badges'
import { Icon, type IconName } from '../components/icons'
import { href } from '../router'

function StatTile({ icon, label, value, note }: { icon: IconName; label: string; value: string | number; note?: string }) {
  return (
    <div className="min-w-0 rounded-2xl bg-white p-4 shadow-sm ring-1 ring-slate-200 sm:p-5">
      <div className="flex items-center gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-teal-50 text-teal-700 sm:h-10 sm:w-10">
          <Icon name={icon} />
        </span>
        <span className="text-sm font-medium leading-tight text-slate-500">{label}</span>
      </div>
      <div className="mt-3 text-2xl font-bold tabular-nums text-slate-900 sm:text-3xl">{value}</div>
      {note && <div className="mt-1 truncate text-xs text-slate-500">{note}</div>}
    </div>
  )
}

function ModelCard({ model }: { model: ModelSummary }) {
  const ready = model.status === 'ready'
  return (
    <a
      href={href({ page: 'model', id: model.id })}
      className="group flex flex-col rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200 transition hover:shadow-md hover:ring-teal-300"
    >
      <div className="flex items-start justify-between gap-3">
        <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-slate-900 text-teal-300">
          <Icon name="microscope" className="h-6 w-6" />
        </span>
        <StatusBadge status={model.status} />
      </div>
      <h3 className="mt-4 text-lg font-semibold text-slate-900">{model.name}</h3>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <CategoryBadge category={model.category} />
        <span>{model.modality}</span>
      </div>
      <p className="mt-3 flex-1 text-sm text-slate-600">{model.summary}</p>

      <dl className="mt-4 grid grid-cols-3 gap-2 border-t border-slate-100 pt-4 text-center">
        <div>
          <dt className="text-[11px] uppercase tracking-wide text-slate-500">Classes</dt>
          <dd className="mt-0.5 font-semibold tabular-nums text-slate-900">{model.classes.length}</dd>
        </div>
        <div>
          <dt className="text-[11px] uppercase tracking-wide text-slate-500">Accuracy</dt>
          <dd className="mt-0.5 font-semibold tabular-nums text-slate-900">{pct(model.test?.accuracy)}</dd>
        </div>
        <div>
          <dt className="text-[11px] uppercase tracking-wide text-slate-500">Macro F1</dt>
          <dd className="mt-0.5 font-semibold tabular-nums text-slate-900">{pct(model.test?.macro_f1)}</dd>
        </div>
      </dl>

      <span className="mt-4 inline-flex items-center gap-1.5 text-sm font-semibold text-teal-700 group-hover:text-teal-800">
        {ready ? 'Open model' : 'View details'}
        <Icon name="arrow" className="h-4 w-4 transition group-hover:translate-x-0.5" />
      </span>
    </a>
  )
}

function AddModelCard() {
  return (
    <div className="flex flex-col rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50/60 p-5">
      <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-white text-slate-500 ring-1 ring-slate-200">
        <Icon name="plus" className="h-6 w-6" />
      </span>
      <h3 className="mt-4 text-lg font-semibold text-slate-900">Add a detection model</h3>
      <ol className="mt-3 list-decimal space-y-1.5 pl-5 text-sm text-slate-600">
        <li>Register it in <code className="rounded bg-white px-1 py-0.5 text-xs ring-1 ring-slate-200">models.yaml</code> with its classes.</li>
        <li>Train it and export to <code className="rounded bg-white px-1 py-0.5 text-xs ring-1 ring-slate-200">artifacts/models/&lt;id&gt;/</code>.</li>
        <li>It appears here automatically.</li>
      </ol>
    </div>
  )
}

export default function DashboardPage({ user, models, loading }: { user: User; models: ModelSummary[]; loading: boolean }) {
  const [userCount, setUserCount] = useState<number | null>(null)
  const isAdmin = user.role === 'admin'

  useEffect(() => {
    if (isAdmin) listUsers().then((u) => setUserCount(u.filter((x) => x.is_active).length)).catch(() => {})
  }, [isAdmin])

  const ready = models.filter((m) => m.status === 'ready').length
  const specialties = new Set(models.map((m) => m.category)).size
  const firstName = (user.full_name || user.username).split(/\s+/)[0]

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Welcome back, {firstName}</h1>
        <p className="mt-1 text-sm text-slate-500">Choose a detection model to analyse an image, or review the models available on the platform.</p>
      </div>

      <div className={`grid grid-cols-2 gap-3 sm:gap-4 ${isAdmin ? 'lg:grid-cols-4' : 'lg:grid-cols-3'}`}>
        <StatTile icon="layers" label="Detection models" value={models.length} note="Registered on the platform" />
        <StatTile icon="check" label="Ready to use" value={ready} note={ready === models.length ? 'All models trained' : `${models.length - ready} awaiting training`} />
        <StatTile icon="heart" label="Specialties" value={specialties} note={[...new Set(models.map((m) => m.category))].join(', ') || '–'} />
        {isAdmin && <StatTile icon="users" label="Active users" value={userCount ?? '–'} note="Accounts that can sign in" />}
      </div>

      <section>
        <div className="mb-4 flex items-end justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Detection models</h2>
            <p className="text-sm text-slate-500">Each model analyses one type of medical image.</p>
          </div>
        </div>
        {loading && models.length === 0 ? (
          <div className="rounded-2xl bg-white p-10 text-center text-sm text-slate-500 ring-1 ring-slate-200">Loading models…</div>
        ) : (
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {models.map((m) => <ModelCard key={m.id} model={m} />)}
            {isAdmin && <AddModelCard />}
          </div>
        )}
      </section>

      <p className="text-xs text-slate-500">
        CellScope ML models are research and screening aids, not diagnostic devices. Every result must be reviewed by a qualified clinician.
      </p>
    </div>
  )
}
