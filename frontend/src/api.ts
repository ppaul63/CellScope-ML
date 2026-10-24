export type RiskLevel = 'Low' | 'Moderate' | 'High'

export interface Prediction {
  model_id: string
  predicted_class: string
  class_name: string
  description: string
  confidence: number
  probabilities: Record<string, number>
  risk_level: RiskLevel
  heatmap_png_base64: string | null
  inference_ms: number
}

export interface ClassInfo {
  code: string
  name: string
  description: string
  risk: RiskLevel | string
}

export type ModelMetrics = Partial<Record<'accuracy' | 'macro_f1' | 'roc_auc_ovr' | 'abnormal_sensitivity' | 'abnormal_specificity' | 'high_grade_recall' | 'test_size', number | null>>

export interface ModelSummary {
  id: string
  name: string
  category: string
  modality: string
  summary: string
  input_hint: string
  status: 'ready' | 'not_trained'
  classes: ClassInfo[]
  arch: string | null
  model_kind: string | null
  image_size: number | null
  heatmap_available: boolean
  test: ModelMetrics | null
}

export type RunStatus = 'running' | 'completed' | 'early_stopped' | 'stopped' | 'interrupted' | 'failed'

export interface EpochMetrics {
  epoch: number
  train_loss: number
  val_loss: number
  val_acc: number
  val_macro_f1: number
  lr?: number
  seconds?: number
}

export interface TrainingRun {
  run_id: string
  model_id: string
  model_name: string
  kind: string
  status: RunStatus
  message: string | null
  arch: string
  device: string
  started_at: string
  updated_at: string
  finished_at: string | null
  epoch: number
  current_epoch?: number
  batch: number
  epochs_total: number
  batches_per_epoch: number
  batch_size: number
  lr: number
  train_size: number
  val_size: number
  patience: number | null
  checkpoint: string
  best: { epoch: number; val_macro_f1: number; val_acc: number } | null
  progress: number
  elapsed_seconds: number
  avg_epoch_seconds: number | null
  eta_seconds: number | null
  latest?: EpochMetrics
  history?: EpochMetrics[]
}

export type Role = 'admin' | 'user'

export interface User {
  id: number
  username: string
  full_name: string
  role: Role
  is_active: boolean
  created_at: string
  last_login_at: string | null
}

const BASE = import.meta.env.VITE_API_URL ?? ''

// The bearer token lives in sessionStorage: it survives a page refresh but is
// dropped when the tab closes, and is never sent to other sites automatically.
const TOKEN_KEY = 'cellscope_token'

function safeStorage(): Storage | null {
  try {
    return window.sessionStorage
  } catch {
    return null
  }
}

let memoryToken: string | null = null
export function getToken(): string | null {
  return safeStorage()?.getItem(TOKEN_KEY) ?? memoryToken
}
function setToken(token: string | null) {
  memoryToken = token
  const store = safeStorage()
  if (!store) return
  if (token) store.setItem(TOKEN_KEY, token)
  else store.removeItem(TOKEN_KEY)
}

// App registers this so any expired/invalid token sends the person back to the login page
let onUnauthorized: () => void = () => {}
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg
  } catch {
    /* not JSON */
  }
  return `Request failed (${res.status})`
}

async function request<T>(path: string, init: RequestInit = {}, { redirectOn401 = true } = {}): Promise<T> {
  const headers = new Headers(init.headers)
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const res = await fetch(`${BASE}${path}`, { ...init, headers })
  if (res.status === 401) {
    setToken(null)
    if (redirectOn401) onUnauthorized()
  }
  if (!res.ok) throw new Error(await errorMessage(res))
  return (res.status === 204 ? undefined : await res.json()) as T
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

// ---- detection models ----
export const listModels = () => request<ModelSummary[]>('/api/models')
export const getModel = (id: string) => request<ModelSummary>(`/api/models/${encodeURIComponent(id)}`)

export function predict(modelId: string, file: File): Promise<Prediction> {
  const form = new FormData()
  form.append('file', file)
  return request(`/api/models/${encodeURIComponent(modelId)}/predict`, { method: 'POST', body: form })
}

// ---- training (admin) ----
export const listTrainingRuns = () => request<TrainingRun[]>('/api/training/runs')
export const getTrainingRun = (modelId: string, runId: string) =>
  request<TrainingRun>(`/api/training/runs/${encodeURIComponent(modelId)}/${encodeURIComponent(runId)}`)

// ---- auth ----
interface TokenResponse {
  access_token: string
  token_type: 'bearer'
  expires_at: string
  user: User
}

export async function login(username: string, password: string, role: Role): Promise<User> {
  const res = await request<TokenResponse>('/api/auth/login', json('POST', { username, password, role }), { redirectOn401: false })
  setToken(res.access_token)
  return res.user
}

export async function logout(): Promise<void> {
  try {
    await request<void>('/api/auth/logout', { method: 'POST' }, { redirectOn401: false })
  } finally {
    setToken(null)
  }
}

export async function getMe(): Promise<User | null> {
  if (!getToken()) return null
  return request<User>('/api/auth/me', {}, { redirectOn401: false })
}
export const changePassword = (current_password: string, new_password: string) =>
  request<void>('/api/auth/change-password', json('POST', { current_password, new_password }))

// ---- user management (admin) ----
export const listUsers = () => request<User[]>('/api/users')
export const createUser = (u: { username: string; full_name: string; password: string; role: Role }) =>
  request<User>('/api/users', json('POST', u))
export const updateUser = (id: number, patch: { is_active?: boolean; role?: Role; password?: string }) =>
  request<User>(`/api/users/${id}`, json('PATCH', patch))
export const deleteUser = (id: number) => request<void>(`/api/users/${id}`, { method: 'DELETE' })
