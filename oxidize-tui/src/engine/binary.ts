/** Locating the `oxidize` binary this TUI should drive. */
import { existsSync, statSync } from "node:fs"
import { dirname, join, resolve } from "node:path"

export interface Launcher {
  /** argv[0] */
  cmd: string
  /** args that must precede user args (e.g. `run -q -p oxidize-cli --`) */
  prefix: string[]
  /** short human label for the status bar */
  label: string
  /** working directory for the child */
  cwd: string
}

/** Walk up from `start` looking for the oxidize workspace root. */
export function findWorkspaceRoot(start = process.cwd()): string | null {
  let dir = resolve(start)
  for (;;) {
    if (existsSync(join(dir, "oxidize-core", "Cargo.toml"))) return dir
    const up = dirname(dir)
    if (up === dir) return null
    dir = up
  }
}

function isExec(p: string): boolean {
  try {
    return statSync(p).isFile()
  } catch {
    return false
  }
}

function onPath(name: string): string | null {
  for (const dir of (process.env.PATH ?? "").split(":")) {
    if (!dir) continue
    const p = join(dir, name)
    if (isExec(p)) return p
  }
  return null
}

export function resolveLauncher(): Launcher {
  const root = findWorkspaceRoot()
  const cwd = root ?? process.cwd()

  const explicit = process.env.OXIDIZE_BIN
  if (explicit && isExec(explicit)) {
    return { cmd: explicit, prefix: [], label: "OXIDIZE_BIN", cwd }
  }

  if (root) {
    for (const profile of ["release", "debug"]) {
      const p = join(root, "target", profile, "oxidize")
      if (isExec(p)) return { cmd: p, prefix: [], label: `target/${profile}`, cwd }
    }
  }

  const found = onPath("oxidize")
  if (found) return { cmd: found, prefix: [], label: "PATH", cwd }

  if (root) {
    return {
      cmd: "cargo",
      prefix: ["run", "-q", "--release", "-p", "oxidize-cli", "--"],
      label: "cargo run (will compile)",
      cwd: root,
    }
  }

  return { cmd: "oxidize", prefix: [], label: "oxidize (not found)", cwd }
}
