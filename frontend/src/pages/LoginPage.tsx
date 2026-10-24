import { useState, type FormEvent } from 'react'
import { login, type Role, type User } from '../api'
import { Alert, Field } from '../components/ui'

export default function LoginPage({ onLogin }: { onLogin: (user: User) => void }) {
  const [role, setRole] = useState<Role>('user')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      onLogin(await login(username, password, role))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign-in failed')
      setPassword('')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4 py-10">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex flex-col items-center text-center">
          <img src="/favicon.svg" alt="" className="h-12 w-12" />
          <h1 className="mt-3 text-2xl font-bold text-slate-900">CellScope ML</h1>
          <p className="text-sm text-slate-500">Health detection platform</p>
        </div>

        <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
          <div className="grid grid-cols-2 rounded-lg bg-slate-100 p-1 text-sm" role="tablist" aria-label="Login type">
            {(['user', 'admin'] as const).map((r) => (
              <button
                key={r}
                type="button"
                role="tab"
                aria-selected={role === r}
                onClick={() => {
                  setRole(r)
                  setError(null)
                }}
                className={`rounded-md py-2 font-semibold transition ${role === r ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                {r === 'user' ? 'User login' : 'Admin login'}
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="mt-5 space-y-4">
            <Field
              label="Username"
              name="username"
              autoComplete="username"
              autoFocus
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
            <Field
              label="Password"
              name="password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            {error && <Alert>{error}</Alert>}
            <button
              type="submit"
              disabled={busy || !username || !password}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-teal-600 px-4 py-2.5 font-semibold text-white shadow-sm transition hover:bg-teal-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {busy && <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />}
              {role === 'admin' ? 'Sign in as administrator' : 'Sign in'}
            </button>
          </form>
          <p className="mt-4 text-center text-xs text-slate-500">
            {role === 'admin' ? 'Administrators can add and manage user accounts.' : 'No account? Ask your administrator to create one.'}
          </p>
        </div>
      </div>
    </div>
  )
}
