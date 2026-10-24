import type { ModelSummary, User } from '../api'
import { href, type Route } from '../router'
import { Icon, type IconName } from './icons'

interface Props {
  user: User
  models: ModelSummary[]
  route: Route
  onNavigate: () => void
  onChangePassword: () => void
  onSignOut: () => void
}

function NavLink({ to, active, icon, label, children, onClick }: {
  to: string
  active: boolean
  icon?: IconName
  label: string
  children?: React.ReactNode
  onClick: () => void
}) {
  return (
    <a
      href={to}
      onClick={onClick}
      aria-current={active ? 'page' : undefined}
      className={`group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${
        active ? 'bg-teal-500/15 text-white' : 'text-slate-300 hover:bg-white/5 hover:text-white'
      }`}
    >
      {icon && <Icon name={icon} className={`h-5 w-5 shrink-0 ${active ? 'text-teal-300' : 'text-slate-400 group-hover:text-slate-200'}`} />}
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {children}
    </a>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <p className="px-3 pb-1 pt-5 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{children}</p>
}

export default function Sidebar({ user, models, route, onNavigate, onChangePassword, onSignOut }: Props) {
  const initials = (user.full_name || user.username).split(/\s+/).map((w) => w[0]).slice(0, 2).join('').toUpperCase()

  return (
    <div className="flex h-full flex-col bg-slate-900">
      <a href={href({ page: 'dashboard' })} onClick={onNavigate} className="flex items-center gap-3 px-5 py-5">
        <img src="/favicon.svg" alt="" className="h-9 w-9" />
        <div className="leading-tight">
          <div className="text-base font-bold text-white">CellScope ML</div>
          <div className="text-xs text-slate-400">Health detection platform</div>
        </div>
      </a>

      <nav className="flex-1 overflow-y-auto px-3 pb-4" aria-label="Main">
        <SectionLabel>Overview</SectionLabel>
        <NavLink to={href({ page: 'dashboard' })} active={route.page === 'dashboard'} icon="dashboard" label="Dashboard" onClick={onNavigate} />

        <SectionLabel>Detection models</SectionLabel>
        {models.map((m) => (
          <NavLink
            key={m.id}
            to={href({ page: 'model', id: m.id })}
            active={route.page === 'model' && route.id === m.id}
            icon="microscope"
            label={m.name}
            onClick={onNavigate}
          >
            <span
              className={`h-2 w-2 shrink-0 rounded-full ${m.status === 'ready' ? 'bg-emerald-400' : 'bg-slate-600'}`}
              title={m.status === 'ready' ? 'Ready' : 'Not trained yet'}
              aria-label={m.status === 'ready' ? 'Ready' : 'Not trained yet'}
            />
          </NavLink>
        ))}
        {models.length === 0 && <p className="px-3 py-2 text-xs text-slate-500">No models registered.</p>}

        {user.role === 'admin' && (
          <>
            <SectionLabel>Administration</SectionLabel>
            <NavLink to={href({ page: 'training' })} active={route.page === 'training'} icon="chart" label="Training" onClick={onNavigate} />
            <NavLink to={href({ page: 'users' })} active={route.page === 'users'} icon="users" label="Users" onClick={onNavigate} />
          </>
        )}
      </nav>

      <div className="border-t border-white/10 p-3">
        <div className="flex items-center gap-3 px-2 py-2">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-teal-600 text-sm font-semibold text-white">{initials}</span>
          <div className="min-w-0 leading-tight">
            <div className="truncate text-sm font-semibold text-white">{user.full_name || user.username}</div>
            <div className="text-xs text-slate-400">{user.role === 'admin' ? 'Administrator' : 'User'}</div>
          </div>
        </div>
        <button type="button" onClick={onChangePassword} className="mt-1 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-slate-300 transition hover:bg-white/5 hover:text-white">
          <Icon name="key" className="h-5 w-5 text-slate-400" /> Change password
        </button>
        <button type="button" onClick={onSignOut} className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-slate-300 transition hover:bg-white/5 hover:text-white">
          <Icon name="logout" className="h-5 w-5 text-slate-400" /> Sign out
        </button>
      </div>
    </div>
  )
}
