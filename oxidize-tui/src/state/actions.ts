/** Everything that talks to the engine. UI components only call these. */
import { basename } from "node:path"

import { listModelIds, streamChat, type ChatMessage } from "../engine/api.js"
import { scrape } from "../engine/metrics.js"
import { enrich, scanModels } from "../engine/models.js"
import { probe, startServer, type ServerHandle } from "../engine/server.js"
import { resolveLauncher } from "../engine/binary.js"
import { getState, nextId, set, setIn, toast, type Message } from "./store.js"

const MAX_LOG_LINES = 2000
const MAX_METRIC_SAMPLES = 240

let handle: ServerHandle | null = null
let startAbort: AbortController | null = null
let genAbort: AbortController | null = null

export function log(stream: "out" | "err" | "tui", text: string) {
  setIn("logs", (l) => {
    const lines = l.lines.length >= MAX_LOG_LINES ? l.lines.slice(-MAX_LOG_LINES + 1) : l.lines.slice()
    lines.push({ id: nextId(), stream, text, at: Date.now() })
    return { lines }
  })
}

// ---------------------------------------------------------------- models ---

export async function rescanModels() {
  setIn("models", { scanning: true })
  const entries = await scanModels()
  setIn("models", (m) => ({
    entries,
    scanning: false,
    cursor: Math.min(m.cursor, Math.max(entries.length - 1, 0)),
  }))
  await enrich(entries, (path, facts, error) => {
    setIn("models", (m) => ({
      entries: m.entries.map((e) => (e.path === path ? { ...e, facts, factsError: error } : e)),
    }))
  })
}

/** `oxidize pull <repo>` into the HF cache, streaming progress into the log view. */
export async function pullModel(repo: string, file?: string) {
  const spec = repo.trim()
  if (!spec) return
  const launcher = resolveLauncher()
  const args = [...launcher.prefix, "pull", spec, ...(file ? ["--file", file] : [])]
  log("tui", `$ ${launcher.cmd} ${args.join(" ")}`)
  set({ view: "logs" })
  toast(`pulling ${spec}…`)

  const proc = Bun.spawn([launcher.cmd, ...args], {
    cwd: launcher.cwd,
    stdout: "pipe",
    stderr: "pipe",
    stdin: "ignore",
  })
  const drain = async (stream: ReadableStream<Uint8Array>, kind: "out" | "err") => {
    for await (const chunk of stream as unknown as AsyncIterable<Uint8Array>) {
      for (const line of new TextDecoder().decode(chunk).split("\n")) {
        if (line.trim()) log(kind, line.replace(/\r$/, ""))
      }
    }
  }
  await Promise.all([
    drain(proc.stdout as ReadableStream<Uint8Array>, "out"),
    drain(proc.stderr as ReadableStream<Uint8Array>, "err"),
  ])
  const code = await proc.exited
  if (code === 0) {
    toast(`pulled ${spec}`, "ok")
    await rescanModels()
  } else {
    toast(`pull failed (exit ${code})`, "err")
  }
}

// ---------------------------------------------------------------- server ---

export function stopServer(quiet = false) {
  startAbort?.abort()
  startAbort = null
  cancelGeneration()
  if (handle) {
    handle.kill()
    handle = null
  }
  stopMetrics()
  setIn("server", { status: "stopped", url: null, pid: null, startedAt: null, detail: "" })
  if (!quiet) toast("server stopped")
}

export async function attach(url: string) {
  const clean = url.replace(/\/+$/, "")
  setIn("server", { status: "starting", url: clean, external: true, detail: `attaching to ${clean}` })
  const status = await probe(clean, 3000)
  if (status === "down") {
    setIn("server", { status: "error", detail: `no server at ${clean}` })
    toast(`no server at ${clean}`, "err")
    return
  }
  let modelId = "oxidize-default"
  try {
    modelId = (await listModelIds(clean))[0] ?? modelId
  } catch {}
  setIn("server", {
    status: status === "ready" ? "ready" : "loading",
    url: clean,
    modelId,
    modelPath: modelId,
    external: true,
    startedAt: Date.now(),
    detail: "",
  })
  log("tui", `attached to ${clean} (model ${modelId})`)
  startMetrics()
}

export async function loadModel(path: string) {
  if (getState().server.external) {
    toast("attached to an external server; detach first", "err")
    return
  }
  stopServer(true)
  const { backend, threads, ctxSize } = getState().server
  const name = basename(path)
  startAbort = new AbortController()

  setIn("server", {
    status: "starting",
    modelPath: path,
    modelId: name.replace(/\.gguf$/i, ""),
    detail: "spawning",
    external: false,
    startedAt: Date.now(),
  })
  set({ view: "chat" })
  log("tui", `loading ${path}`)

  try {
    setIn("server", { status: "loading", detail: "loading weights" })
    handle = await startServer(
      { model: path, backend, threads, ctxSize, maxTokens: getState().params.maxTokens },
      log,
      startAbort.signal,
    )
    let modelId = getState().server.modelId
    try {
      modelId = (await listModelIds(handle.url))[0] ?? modelId
    } catch {}
    setIn("server", {
      status: "ready",
      url: handle.url,
      pid: handle.pid ?? null,
      modelId,
      detail: "",
    })
    toast(`${name} ready`, "ok")
    log("tui", `ready on ${handle.url}`)
    startMetrics()
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err)
    if (detail === "cancelled") return
    setIn("server", { status: "error", detail })
    log("err", detail)
    toast(detail, "err")
  }
}

export function restartServer() {
  const path = getState().server.modelPath
  if (!path || getState().server.external) {
    toast("nothing to restart", "err")
    return
  }
  void loadModel(path)
}

// --------------------------------------------------------------- metrics ---

let metricsTimer: ReturnType<typeof setInterval> | null = null

export function startMetrics() {
  stopMetrics()
  metricsTimer = setInterval(async () => {
    const url = getState().server.url
    if (!url) return
    try {
      const snap = await scrape(url)
      setIn("metrics", (m) => ({
        latest: snap,
        error: null,
        history: [...m.history, snap].slice(-MAX_METRIC_SAMPLES),
      }))
    } catch (err) {
      setIn("metrics", { error: err instanceof Error ? err.message : String(err) })
    }
  }, 1000)
}

export function stopMetrics() {
  if (metricsTimer) clearInterval(metricsTimer)
  metricsTimer = null
}

// ------------------------------------------------------------------ chat ---

function updateMessage(id: number, patch: Partial<Message>) {
  setIn("chat", (chat) => ({
    messages: chat.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
  }))
}

export function clearChat() {
  cancelGeneration()
  setIn("chat", { messages: [], tps: 0, tpsHistory: [] })
}

export function cancelGeneration() {
  genAbort?.abort()
  genAbort = null
}

export async function send(text: string) {
  const trimmed = text.trim()
  if (!trimmed) return
  const s = getState()
  if (s.chat.generating) return
  if (s.server.status !== "ready" || !s.server.url) {
    toast("no model loaded — press ctrl+t for the model browser", "err")
    return
  }

  const userMsg: Message = { id: nextId(), role: "user", content: trimmed, at: Date.now() }
  const reply: Message = {
    id: nextId(),
    role: "assistant",
    content: "",
    at: Date.now(),
    streaming: true,
    stats: { tokens: 0, elapsed: 0 },
  }
  setIn("chat", (chat) => ({
    messages: [...chat.messages, userMsg, reply],
    generating: true,
    tps: 0,
    tpsHistory: [],
  }))

  const history: ChatMessage[] = []
  if (s.chat.system.trim()) history.push({ role: "system", content: s.chat.system })
  for (const m of [...s.chat.messages, userMsg]) {
    if (m.role === "system" || m.error) continue
    if (!m.content.trim()) continue
    history.push({ role: m.role, content: m.content })
  }

  const started = performance.now()
  let ttft: number | undefined
  let tokens = 0
  let pending = ""
  let lastFlush = 0

  // Coalesce deltas: a fast local model emits faster than the terminal can draw.
  const flush = (force = false) => {
    const now = performance.now()
    if (!force && (pending === "" || now - lastFlush < 40)) return
    lastFlush = now
    const chunk = pending
    pending = ""
    const elapsed = now - started
    const genElapsed = ttft == null ? elapsed : elapsed - ttft
    const tps = genElapsed > 0 ? (tokens / genElapsed) * 1000 : 0
    setIn("chat", (chat) => ({
      messages: chat.messages.map((m) =>
        m.id === reply.id
          ? { ...m, content: m.content + chunk, stats: { ttft, tokens, elapsed } }
          : m,
      ),
      tps,
      tpsHistory: [...chat.tpsHistory, tps].slice(-120),
    }))
  }

  genAbort = new AbortController()
  try {
    await streamChat(
      s.server.url,
      s.server.modelId,
      history,
      s.params,
      {
        onFirstToken: () => {
          ttft = performance.now() - started
        },
        onDelta: (d) => {
          tokens++ // deltas are token-granular for oxidize's streaming path
          pending += d
          flush()
        },
      },
      genAbort.signal,
    )
    flush(true)
    updateMessage(reply.id, { streaming: false })
  } catch (err) {
    flush(true)
    const aborted = err instanceof Error && err.name === "AbortError"
    const detail = err instanceof Error ? err.message : String(err)
    setIn("chat", (chat) => ({
      messages: chat.messages.map((m) =>
        m.id === reply.id
          ? {
              ...m,
              streaming: false,
              error: !aborted,
              content: aborted ? m.content + "\n\n_[cancelled]_" : detail,
            }
          : m,
      ),
    }))
    if (!aborted) log("err", detail)
  } finally {
    genAbort = null
    setIn("chat", { generating: false })
  }
}

export function shutdown() {
  cancelGeneration()
  stopMetrics()
  startAbort?.abort()
  handle?.kill()
}
