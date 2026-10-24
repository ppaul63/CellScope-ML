import { useEffect, useState } from 'react'
import { predict, type ModelSummary, type Prediction } from '../api'
import AboutModel from '../components/AboutModel'
import { CategoryBadge, StatusBadge } from '../components/badges'
import ResultCard from '../components/ResultCard'
import UploadZone from '../components/UploadZone'
import { href } from '../router'

export default function ModelPage({ model }: { model: ModelSummary }) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const ready = model.status === 'ready'

  // Start fresh when switching between models
  useEffect(() => {
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
  }, [model.id])

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const handleFile = (f: File) => {
    setFile(f)
    setResult(null)
    setError(null)
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      setResult(await predict(model.id, file))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <nav className="text-sm text-slate-500" aria-label="Breadcrumb">
          <a href={href({ page: 'dashboard' })} className="hover:text-slate-700">Dashboard</a>
          <span className="mx-2">/</span>
          <span className="text-slate-700">{model.name}</span>
        </nav>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-bold text-slate-900">{model.name}</h1>
          <StatusBadge status={model.status} />
        </div>
        <div className="mt-1.5 flex flex-wrap items-center gap-2 text-sm text-slate-500">
          <CategoryBadge category={model.category} />
          <span>{model.modality}</span>
        </div>
        <p className="mt-3 max-w-3xl text-sm text-slate-600">{model.summary}</p>
      </div>

      <div role="note" className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
        <strong>Research and screening aid only — not a diagnosis.</strong> Results must be reviewed by a qualified
        clinician before any clinical decision.
      </div>

      {!ready && (
        <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700">
          This model hasn't been trained yet, so it can't analyse images. Once it's trained and exported, it becomes available here without a restart.
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <p className="text-sm font-medium text-slate-700">{model.input_hint}</p>
          <UploadZone
            previewUrl={previewUrl}
            fileName={file?.name ?? null}
            disabled={loading || !ready}
            onFile={handleFile}
            onReject={setError}
          />
          <button
            type="button"
            onClick={analyze}
            disabled={!file || loading || !ready}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-teal-600 px-4 py-3 font-semibold text-white shadow-sm transition hover:bg-teal-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading && <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />}
            {loading ? 'Analysing…' : 'Analyse image'}
          </button>
          {error && (
            <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
              {error}
            </div>
          )}
        </div>

        <div>
          {result && previewUrl ? (
            <ResultCard result={result} previewUrl={previewUrl} />
          ) : (
            <div className="flex h-full min-h-72 items-center justify-center rounded-2xl border border-slate-200 bg-white p-6 text-center text-sm text-slate-500">
              {loading ? 'Running the model…' : ready ? 'Upload an image and click “Analyse image” to see the result here.' : 'Results will appear here once the model is trained.'}
            </div>
          )}
        </div>
      </div>

      <AboutModel model={model} />
    </div>
  )
}
