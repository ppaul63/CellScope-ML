import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { createUser, deleteUser, listUsers, updateUser, type Role, type User } from '../api'
import { Alert, Button, Field, Modal } from '../components/ui'

const PASSWORD_HINT = 'At least 10 characters, mixing letters with numbers or symbols.'

function formatDate(ts: string | null) {
  if (!ts) return 'Never'
  return new Date(ts.replace(' ', 'T') + 'Z').toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function AddUserForm({ onCreated }: { onCreated: (u: User) => void }) {
  const empty = { username: '', full_name: '', password: '', role: 'user' as Role }
  const [form, setForm] = useState(empty)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      onCreated(await createUser(form))
      setForm(empty)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create user')
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
      <h2 className="text-lg font-semibold text-slate-900">Add user</h2>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <Field label="Username" required autoComplete="off" value={form.username}
          hint="3–32 letters, numbers, dot, dash or underscore."
          onChange={(e) => setForm({ ...form, username: e.target.value })} />
        <Field label="Full name" autoComplete="off" value={form.full_name}
          onChange={(e) => setForm({ ...form, full_name: e.target.value })} />
        <Field label="Initial password" type="password" required autoComplete="new-password" value={form.password}
          hint={PASSWORD_HINT} onChange={(e) => setForm({ ...form, password: e.target.value })} />
        <label className="block text-sm">
          <span className="font-medium text-slate-700">Role</span>
          <select
            value={form.role}
            onChange={(e) => setForm({ ...form, role: e.target.value as Role })}
            className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 shadow-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/30"
          >
            <option value="user">User: can analyse images</option>
            <option value="admin">Admin: can also manage users</option>
          </select>
        </label>
      </div>
      {error && <div className="mt-4"><Alert>{error}</Alert></div>}
      <div className="mt-4 flex justify-end">
        <Button type="submit" disabled={busy || !form.username || !form.password}>
          {busy ? 'Adding…' : 'Add user'}
        </Button>
      </div>
    </form>
  )
}

function ResetPasswordModal({ user, onClose, onDone }: { user: User; onClose: () => void; onDone: () => void }) {
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    try {
      await updateUser(user.id, { password })
      onDone()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not reset password')
    }
  }

  return (
    <Modal title={`Reset password for ${user.username}`} onClose={onClose}>
      <form onSubmit={submit} className="space-y-4">
        <Field label="New password" type="password" autoFocus required autoComplete="new-password"
          hint={`${PASSWORD_HINT} The user will be signed out everywhere.`}
          value={password} onChange={(e) => setPassword(e.target.value)} />
        {error && <Alert>{error}</Alert>}
        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={onClose}>Cancel</Button>
          <Button type="submit" disabled={!password}>Reset password</Button>
        </div>
      </form>
    </Modal>
  )
}

export default function UsersPage({ me }: { me: User }) {
  const [users, setUsers] = useState<User[]>([])
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [resetting, setResetting] = useState<User | null>(null)

  const refresh = useCallback(() => {
    listUsers().then(setUsers).catch((e) => setError(e.message))
  }, [])
  useEffect(refresh, [refresh])

  const act = async (fn: () => Promise<unknown>, message: string) => {
    setError(null)
    setNotice(null)
    try {
      await fn()
      setNotice(message)
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Action failed')
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Users</h1>
        <p className="text-sm text-slate-500">Add people who can use CellScope ML, and control their access.</p>
      </div>

      <AddUserForm onCreated={(u) => { setNotice(`Added ${u.username}.`); setError(null); refresh() }} />

      {error && <Alert>{error}</Alert>}
      {notice && <Alert kind="success">{notice}</Alert>}

      <section className="overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
        <div className="border-b border-slate-200 px-6 py-4">
          <h2 className="text-lg font-semibold text-slate-900">Accounts <span className="text-sm font-normal text-slate-500">({users.length})</span></h2>
        </div>
        <ul className="divide-y divide-slate-200">
          {users.map((u) => {
            const self = u.id === me.id
            return (
              <li key={u.id} className="flex flex-col gap-3 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold text-slate-900">{u.username}</span>
                    {self && <span className="text-xs text-slate-500">(you)</span>}
                    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${u.role === 'admin' ? 'bg-indigo-100 text-indigo-800' : 'bg-slate-100 text-slate-700'}`}>
                      {u.role === 'admin' ? 'Admin' : 'User'}
                    </span>
                    {!u.is_active && <span className="rounded-full bg-rose-100 px-2 py-0.5 text-xs font-semibold text-rose-800">Disabled</span>}
                  </div>
                  <p className="mt-0.5 text-sm text-slate-500">
                    {u.full_name && <>{u.full_name} · </>}Last sign-in: {formatDate(u.last_login_at)}
                  </p>
                </div>
                {!self && (
                  <div className="flex flex-wrap gap-2">
                    <Button variant="secondary" onClick={() => setResetting(u)}>Reset password</Button>
                    <Button
                      variant="secondary"
                      onClick={() => act(() => updateUser(u.id, { is_active: !u.is_active }), `${u.username} ${u.is_active ? 'disabled' : 'enabled'}.`)}
                    >
                      {u.is_active ? 'Disable' : 'Enable'}
                    </Button>
                    <Button
                      variant="danger"
                      onClick={() => {
                        if (window.confirm(`Delete ${u.username}? This cannot be undone.`))
                          act(() => deleteUser(u.id), `Deleted ${u.username}.`)
                      }}
                    >
                      Delete
                    </Button>
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      </section>

      {resetting && (
        <ResetPasswordModal
          user={resetting}
          onClose={() => setResetting(null)}
          onDone={() => { setNotice(`Password reset for ${resetting.username}.`); setResetting(null) }}
        />
      )}
    </div>
  )
}
