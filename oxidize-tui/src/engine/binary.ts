/** Locating the C `oxidize-c` runtime this TUI should drive. */
import { existsSync, statSync } from "node:fs"
import { dirname, join, resolve } from "node:path"

export interface Launcher {
  /** argv[0] */
  cmd: string
  /** args that must precede user args (e.g. `make -C oxidize-c --`) */
  prefix: string[]
  /** short human label for the status bar */
  label: string
  /** working directory for the child */
  cwd: string
}

function isWorkspaceRoot(dir: string): boolean {
  return existsSync(join(dir, "oxidize-c", "Makefile")) && existsSync(join(dir, "oxidize-tui", "package.json"))
}

/** Walk up from `start` looking for the oxidize workspace root. */
export function findWorkspaceRoot(start = process.cwd()): string | null {
  let dir = resolve(start)
  for (;;) {
    if (isWorkspaceRoot(dir)) return dir
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
    return { cmd: resolve(explicit), prefix: [], label: "OXIDIZE_BIN", cwd }
  }

  if (root) {
    const built = join(root, "oxidize-c", "oxidize-c")
    if (isExec(built)) return { cmd: built, prefix: [], label: "oxidize-c", cwd }
  }

  const foundC = onPath("oxidize-c")
  if (foundC) return { cmd: foundC, prefix: [], label: "PATH", cwd }

  if (root) {
    return {
      cmd: join(root, "oxidize-c", "oxidize-c"),
      prefix: [],
      label: "oxidize-c (run make first)",
      cwd: root,
    }
  }

  return { cmd: "oxidize-c", prefix: [], label: "oxidize-c (not found)", cwd }
}
