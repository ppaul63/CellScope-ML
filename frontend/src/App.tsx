import { useCallback, useEffect, useState } from 'react'
import { getMe, listModels, logout, setUnauthorizedHandler, type ModelSummary, type User } from './api'
import ChangePasswordModal from './components/ChangePasswordModal'
import { Icon } from './components/icons'
import Sidebar from './components/Sidebar'
import DashboardPage from './pages/DashboardPage'
import LoginPage from './pages/LoginPage'
import ModelPage from './pages/ModelPage'
import TrainingPage from './pages/TrainingPage'
import UsersPage from './pages/UsersPage'
import { href, navigate, useRoute } from './router'

export default function App() {
  const [user, setUser] = useState<User | null>(null)
  const [checking, setChecking] = useState(true)
  const [models, setModels] = useState<ModelSummary[]>([])
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [changingPassword, setChangingPassword] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const route = useRoute()

  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null))
    getMe()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setChecking(false))
  }, [])

  const refreshModels = useCallback(() => {
    setModelsLoading(true)
    listModels()
      .then((m) => { setModels(m); setModelsError(null) })
      .catch((e) => setModelsError(e instanceof Error ? e.message : String(e)))
      .finally(() => setModelsLoading(false))
  }, [])

  // Load the model list after sign-in, and reload it whenever the dashboard is opened
  // so newly trained models show up without a page refresh.
  const onDashboard = route.page === 'dashboard'
  useEffect(() => {
    if (user) refreshModels()
  }, [user, onDashboard, refreshModels])

  if (checking) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <span className="h-6 w-6 animate-spin rounded-full border-2 border-teal-600/30 border-t-teal-600" aria-label="Loading" />
      </div>
    )
  }

  if (!user) {
    return <LoginPage onLogin={(u) => { setUser(u); navigate({ page: 'dashboard' }) }} />
  }

  const signOut = async () => {
    await logout().catch(() => {})
    setUser(null)
    setModels([])
    navigate({ page: 'dashboard' })
  }

  let page: React.ReactNode
  if (route.page === 'users' && user.role === 'admin') {
    page = <UsersPage me={user} />
  } else if (route.page === 'training' && user.role === 'admin') {
    page = <TrainingPage modelId={route.modelId} runId={route.runId} />
  } else if (route.page === 'model') {
    const model = models.find((m) => m.id === route.id)
    page = model ? (
      <ModelPage model={model} />
    ) : (
      <div className="rounded-2xl bg-white p-10 text-center text-sm text-slate-500 ring-1 ring-slate-200">
        {modelsLoading ? 'Loading model…' : <>Model not found. <a className="font-semibold text-teal-700" href={href({ page: 'dashboard' })}>Back to dashboard</a></>}
      </div>
    )
  } else {
    page = <DashboardPage user={user} models={models} loading={modelsLoading} />
  }

  const sidebar = (
    <Sidebar
      user={user}
      models={models}
      route={route}
      onNavigate={() => setSidebarOpen(false)}
      onChangePassword={() => { setSidebarOpen(false); setChangingPassword(true) }}
      onSignOut={signOut}
    />
  )

  return (
    <div className="min-h-screen">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 lg:block">{sidebar}</aside>

      {/* Mobile drawer */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-slate-900/50" onClick={() => setSidebarOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 max-w-[85%] shadow-xl">
            <button type="button" onClick={() => setSidebarOpen(false)} className="absolute right-3 top-5 rounded-lg p-1 text-slate-400 hover:text-white" aria-label="Close menu">
              <Icon name="close" />
            </button>
            {sidebar}
          </aside>
        </div>
      )}

      <div className="lg:pl-64">
        {/* Mobile top bar */}
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-slate-200 bg-white px-4 py-3 lg:hidden">
          <button type="button" onClick={() => setSidebarOpen(true)} className="rounded-lg p-1.5 text-slate-600 hover:bg-slate-100" aria-label="Open menu">
            <Icon name="menu" className="h-6 w-6" />
          </button>
          <img src="/favicon.svg" alt="" className="h-7 w-7" />
          <span className="font-bold text-slate-900">CellScope ML</span>
        </header>

        <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
          {modelsError && (
            <div role="alert" className="mb-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
              Couldn't load models: {modelsError}
            </div>
          )}
          {page}
        </main>
      </div>

      {changingPassword && <ChangePasswordModal onClose={() => setChangingPassword(false)} />}
    </div>
  )
}
