/** Spawns and supervises `oxidize serve`, or attaches to a running instance. */
import { createServer } from "node:net"

import { resolveLauncher, type Launcher } from "./binary.js"

export interface ServeOptions {
  model: string
  backend?: string
  threads?: number
  ctxSize?: number
  maxTokens?: number
  extraArgs?: string[]
  host?: string
  port?: number
}

export interface ServerHandle {
  url: string
  pid?: number
  kill: () => void
}

type LogSink = (stream: "out" | "err" | "tui", line: string) => void

export function freePort(): Promise<number> {
  return new Promise((res, rej) => {
    const srv = createServer()
    srv.on("error", rej)
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address()
      if (addr && typeof addr === "object") {
        const p = addr.port
        srv.close(() => res(p))
      } else {
        srv.close(() => rej(new Error("could not allocate a port")))
      }
    })
  })
}

export async function probe(url: string, timeoutMs = 1500): Promise<"ready" | "loading" | "down"> {
  const ctl = AbortSignal.timeout(timeoutMs)
  try {
    const r = await fetch(`${url}/readyz`, { signal: ctl })
    if (r.status === 200) return "ready"
    return "loading"
  } catch {
    return "down"
  }
}

function pumpLines(stream: ReadableStream<Uint8Array> | null, emit: (line: string) => void) {
  if (!stream) return
  const reader = stream.getReader()
  const dec = new TextDecoder()
  let buf = ""
  const loop = async () => {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const parts = buf.split("\n")
      buf = parts.pop() ?? ""
      for (const p of parts) if (p.trim()) emit(p.replace(/\r$/, ""))
    }
    if (buf.trim()) emit(buf)
  }
  loop().catch(() => {})
}

export function buildArgs(launcher: Launcher, opts: ServeOptions, host: string, port: number) {
  const args = [...launcher.prefix, "serve", opts.model, "--host", host, "--port", String(port)]
  if (opts.backend && opts.backend !== "auto") args.push("--backend", opts.backend)
  if (opts.threads && opts.threads > 0) args.push("--threads", String(opts.threads))
  if (opts.ctxSize && opts.ctxSize > 0) args.push("--ctx-size", String(opts.ctxSize))
  if (opts.maxTokens && opts.maxTokens > 0) args.push("--max-tokens", String(opts.maxTokens))
  if (opts.extraArgs?.length) args.push(...opts.extraArgs)
  return args
}

/**
 * Start a server and resolve once `/readyz` reports 200. Model load for a large
 * GGUF can take minutes, so there is no hard deadline — the caller cancels.
 */
export async function startServer(
  opts: ServeOptions,
  log: LogSink,
  signal: AbortSignal,
): Promise<ServerHandle> {
  const launcher = resolveLauncher()
  const host = opts.host ?? "127.0.0.1"
  const port = opts.port ?? (await freePort())
  const args = buildArgs(launcher, opts, host, port)

  log("tui", `$ ${launcher.cmd} ${args.join(" ")}`)
  const proc = Bun.spawn([launcher.cmd, ...args], {
    cwd: launcher.cwd,
    stdout: "pipe",
    stderr: "pipe",
    stdin: "ignore",
    env: { ...process.env, RUST_LOG: process.env.RUST_LOG ?? "info" },
  })

  pumpLines(proc.stdout as ReadableStream<Uint8Array>, (l) => log("out", l))
  pumpLines(proc.stderr as ReadableStream<Uint8Array>, (l) => log("err", l))

  const url = `http://${host}:${port}`
  const kill = () => {
    try {
      proc.kill("SIGTERM")
    } catch {}
    setTimeout(() => {
      try {
        proc.kill("SIGKILL")
      } catch {}
    }, 4000).unref?.()
  }

  let exited = false
  proc.exited.then((code) => {
    exited = true
    log("tui", `server exited with code ${code}`)
  })

  signal.addEventListener("abort", kill, { once: true })

  for (;;) {
    if (signal.aborted) {
      kill()
      throw new Error("cancelled")
    }
    if (exited) throw new Error("server process exited before becoming ready")
    if ((await probe(url, 900)) === "ready") break
    await Bun.sleep(300)
  }

  return { url, pid: proc.pid, kill }
}
