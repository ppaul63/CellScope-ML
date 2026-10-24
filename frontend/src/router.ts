import { useEffect, useState } from 'react'

// Tiny hash router: #/ (dashboard), #/models/<id>, #/training[/<model>/<run>], #/users.
// Hash routes survive a refresh without any server-side routing config.
export type Route =
  | { page: 'dashboard' }
  | { page: 'model'; id: string }
  | { page: 'training'; modelId?: string; runId?: string }
  | { page: 'users' }

function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean)
  if (parts[0] === 'models' && parts[1]) return { page: 'model', id: decodeURIComponent(parts[1]) }
  if (parts[0] === 'users') return { page: 'users' }
  if (parts[0] === 'training') {
    return parts[1] && parts[2]
      ? { page: 'training', modelId: decodeURIComponent(parts[1]), runId: decodeURIComponent(parts[2]) }
      : { page: 'training' }
  }
  return { page: 'dashboard' }
}

export function href(route: Route): string {
  if (route.page === 'model') return `#/models/${encodeURIComponent(route.id)}`
  if (route.page === 'users') return '#/users'
  if (route.page === 'training') {
    return route.modelId && route.runId
      ? `#/training/${encodeURIComponent(route.modelId)}/${encodeURIComponent(route.runId)}`
      : '#/training'
  }
  return '#/'
}

export function navigate(route: Route) {
  window.location.hash = href(route)
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash))
  useEffect(() => {
    const onChange = () => {
      setRoute(parse(window.location.hash))
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}
