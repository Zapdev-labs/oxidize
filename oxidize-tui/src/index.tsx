#!/usr/bin/env bun
import { createCliRenderer } from "@opentui/core"
import { createRoot } from "@opentui/react"
import { resolve } from "node:path"

import { App } from "./App.js"
import { resolveLauncher } from "./engine/binary.js"
import { attach, loadModel, log, rescanModels, shutdown } from "./state/actions.js"
import { set, setIn } from "./state/store.js"

const USAGE = `oxidize-tui — terminal console for the oxidize inference engine

Usage:
  oxidize-tui [model.gguf] [options]

Options:
  --api URL         attach to an already-running oxidize server instead of spawning one
  --backend NAME    cpu | cuda | metal | vulkan | rocm | mlx        (default: cpu)
  --threads N       worker threads for the spawned server           (default: engine default)
  --ctx-size N      context window for the spawned server
  --max-tokens N    default completion budget                       (default: 512)
  --temperature F   sampling temperature                            (default: 0.8)
  --top-p F         nucleus sampling                                (default: 0.95)
  --system TEXT     system prompt prepended to every conversation
  -h, --help        this message

Environment:
  OXIDIZE_BIN       path to the oxidize binary (else target/release, target/debug, then PATH)
  OXIDIZE_MODELS    extra ':'-separated directories to scan for .gguf files
`

interface Cli {
  model?: string
  api?: string
  backend: string
  threads: number
  ctxSize: number
  maxTokens: number
  temperature: number
  topP: number
  system: string
}

function parseArgs(argv: string[]): Cli {
  const cli: Cli = {
    backend: "cpu",
    threads: 0,
    ctxSize: 0,
    maxTokens: 512,
    temperature: 0.8,
    topP: 0.95,
    system: "",
  }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!
    const next = () => {
      const v = argv[++i]
      if (v == null) {
        process.stderr.write(`${a} requires a value\n`)
        process.exit(2)
      }
      return v
    }
    switch (a) {
      case "-h":
      case "--help":
        process.stdout.write(USAGE)
        process.exit(0)
      case "--api":
        cli.api = next()
        break
      case "--backend":
        cli.backend = next()
        break
      case "--threads":
        cli.threads = Number(next())
        break
      case "--ctx-size":
        cli.ctxSize = Number(next())
        break
      case "--max-tokens":
        cli.maxTokens = Number(next())
        break
      case "--temperature":
        cli.temperature = Number(next())
        break
      case "--top-p":
        cli.topP = Number(next())
        break
      case "--system":
        cli.system = next()
        break
      default:
        if (a.startsWith("-")) {
          process.stderr.write(`unknown option ${a}\n\n${USAGE}`)
          process.exit(2)
        }
        cli.model = a
    }
  }
  return cli
}

const cli = parseArgs(process.argv.slice(2))

setIn("server", { backend: cli.backend, threads: cli.threads, ctxSize: cli.ctxSize })
setIn("params", { maxTokens: cli.maxTokens, temperature: cli.temperature, topP: cli.topP })
setIn("chat", { system: cli.system })

const renderer = await createCliRenderer({ exitOnCtrlC: false, targetFps: 60 })
createRoot(renderer).render(<App />)

const launcher = resolveLauncher()
log("tui", `engine: ${launcher.cmd} (${launcher.label})`)

if (cli.api) {
  void attach(cli.api)
} else if (cli.model) {
  void loadModel(resolve(cli.model))
} else {
  set({ view: "models" })
}
void rescanModels()

const bye = () => {
  shutdown()
  process.exit(0)
}
process.on("SIGINT", bye)
process.on("SIGTERM", bye)
process.on("exit", () => shutdown())
