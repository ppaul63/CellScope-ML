import { useState, type FormEvent } from 'react'
import { changePassword } from '../api'
import { Alert, Button, Field, Modal } from './ui'

export default function ChangePasswordModal({ onClose }: { onClose: () => void }) {
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (next !== confirm) return setError('New passwords do not match.')
    setError(null)
    try {
      await changePassword(current, next)
      setDone(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not change password')
    }
  }

  return (
    <Modal title="Change password" onClose={onClose}>
      {done ? (
        <div className="space-y-4">
          <Alert kind="success">Password changed. Other devices have been signed out.</Alert>
          <div className="flex justify-end"><Button onClick={onClose}>Done</Button></div>
        </div>
      ) : (
        <form onSubmit={submit} className="space-y-4">
          <Field label="Current password" type="password" autoFocus required autoComplete="current-password"
            value={current} onChange={(e) => setCurrent(e.target.value)} />
          <Field label="New password" type="password" required autoComplete="new-password"
            hint="At least 10 characters, mixing letters with numbers or symbols."
            value={next} onChange={(e) => setNext(e.target.value)} />
          <Field label="Confirm new password" type="password" required autoComplete="new-password"
            value={confirm} onChange={(e) => setConfirm(e.target.value)} />
          {error && <Alert>{error}</Alert>}
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={!current || !next || !confirm}>Change password</Button>
          </div>
        </form>
      )}
    </Modal>
  )
}
