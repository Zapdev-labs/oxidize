/** Tiny immutable external store: `set()` replaces the root, hooks memo per snapshot. */
import { useCallback, useRef, useSyncExternalStore } from "react"

import type { MetricsSnapshot } from "../engine/metrics.js"
import type { ModelEntry } from "../engine/models.js"

export type View = "chat" | "models" | "monitor" | "logs"

export type ServerStatus = "idle" | "starting" | "loading" | "ready" | "error" | "stopped"

export interface Message {
  id: number
  role: "user" | "assistant" | "system"
  content: string
  at: number
  streaming?: boolean
  error?: boolean
  stats?: { ttft?: number; tokens: number; elapsed: number }
}

export interface LogLine {
  id: number
  stream: "out" | "err" | "tui"
  text: string
  at: number
}

export interface State {
  view: View
  ready: boolean

  server: {
    status: ServerStatus
    url: string | null
    pid: number | null
    modelPath: string | null
    modelId: string
    backend: string
    threads: number
    ctxSize: number
    detail: string
    external: boolean
    startedAt: number | null
  }

  models: {
    entries: ModelEntry[]
    scanning: boolean
    cursor: number
    filter: string
    filtering: boolean
    inspect: string | null
  }

  chat: {
    messages: Message[]
    generating: boolean
    draft: string
    system: string
    tps: number
    tpsHistory: number[]
  }

  params: {
    temperature: number
    topP: number
    topK: number
    maxTokens: number
  }

  logs: { lines: LogLine[]; follow: boolean }

  metrics: { latest: MetricsSnapshot | null; history: MetricsSnapshot[]; error: string | null }

  overlay:
    | null
    | { kind: "palette"; query: string; cursor: number }
    | { kind: "help" }
    | { kind: "prompt"; label: string; placeholder: string; submit: (value: string) => void }

  toast: { text: string; tone: "info" | "ok" | "err"; at: number } | null
}

const initial: State = {
  view: "chat",
  ready: false,
  server: {
    status: "idle",
    url: null,
    pid: null,
    modelPath: null,
    modelId: "oxidize-default",
    backend: "cpu",
    threads: 0,
    ctxSize: 0,
    detail: "",
    external: false,
    startedAt: null,
  },
  models: { entries: [], scanning: false, cursor: 0, filter: "", filtering: false, inspect: null },
  chat: {
    messages: [],
    generating: false,
    draft: "",
    system: "",
    tps: 0,
    tpsHistory: [],
  },
  params: { temperature: 0.8, topP: 0.95, topK: 0, maxTokens: 512 },
  logs: { lines: [], follow: true },
  metrics: { latest: null, history: [], error: null },
  overlay: null,
  toast: null,
}

let state = initial
const listeners = new Set<() => void>()

export const getState = () => state

export function set(patch: Partial<State> | ((s: State) => Partial<State>)) {
  const next = typeof patch === "function" ? patch(state) : patch
  state = { ...state, ...next }
  for (const l of listeners) l()
}

/** Patch one top-level slice without clobbering its siblings. */
export function setIn<K extends keyof State>(
  key: K,
  patch: Partial<State[K]> | ((s: State[K]) => Partial<State[K]>),
) {
  const cur = state[key]
  const next = typeof patch === "function" ? (patch as (s: State[K]) => Partial<State[K]>)(cur) : patch
  state = { ...state, [key]: { ...(cur as object), ...(next as object) } as State[K] }
  for (const l of listeners) l()
}

const subscribe = (cb: () => void) => {
  listeners.add(cb)
  return () => listeners.delete(cb)
}

export function useStore<T>(selector: (s: State) => T): T {
  const selectorRef = useRef(selector)
  selectorRef.current = selector
  const memo = useRef<{ snapshot?: State; value?: T }>({})
  // Memoise per snapshot so object-returning selectors stay referentially stable
  // between renders; otherwise useSyncExternalStore would loop forever.
  const get = useCallback(() => {
    if (memo.current.snapshot !== state) {
      memo.current = { snapshot: state, value: selectorRef.current(state) }
    }
    return memo.current.value as T
  }, [])
  return useSyncExternalStore(subscribe, get, get)
}

let seq = 0
export const nextId = () => ++seq

export function toast(text: string, tone: "info" | "ok" | "err" = "info") {
  set({ toast: { text, tone, at: Date.now() } })
}
