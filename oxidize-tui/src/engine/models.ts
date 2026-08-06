/** Local GGUF discovery + lazy header enrichment. */
import { readdir, stat } from "node:fs/promises"
import { homedir } from "node:os"
import { basename, join, resolve } from "node:path"

import { factsFrom, readGgufHeader, type ModelFacts } from "./gguf.js"
import { findWorkspaceRoot } from "./binary.js"

export interface ModelEntry {
  path: string
  name: string
  size: number
  mtime: number
  /** number of shards when the model is split (`-00001-of-000NN.gguf`) */
  shards: number
  source: string
  facts?: ModelFacts
  factsError?: string
}

const SHARD_RE = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i

export function searchRoots(): { dir: string; label: string }[] {
  const roots: { dir: string; label: string }[] = []
  const push = (dir: string, label: string) => {
    const r = resolve(dir)
    if (!roots.some((x) => x.dir === r)) roots.push({ dir: r, label })
  }

  for (const dir of (process.env.OXIDIZE_MODELS ?? "").split(":").filter(Boolean)) {
    push(dir, "env")
  }
  push(join(process.cwd(), "models"), "cwd")
  const root = findWorkspaceRoot()
  if (root) push(join(root, "models"), "workspace")
  push(join(homedir(), ".cache", "oxidize", "hf"), "hf-cache")
  push(join(homedir(), ".cache", "huggingface", "hub"), "hf-hub")
  push(join(homedir(), "models"), "home")
  return roots
}

async function walk(dir: string, depth: number, out: string[], budget: { n: number }) {
  if (depth < 0 || budget.n <= 0) return
  let entries
  try {
    entries = await readdir(dir, { withFileTypes: true })
  } catch {
    return
  }
  for (const e of entries) {
    if (budget.n <= 0) return
    if (e.name.startsWith(".")) continue
    const full = join(dir, e.name)
    if (e.isDirectory()) {
      await walk(full, depth - 1, out, budget)
    } else if (e.isFile() && e.name.toLowerCase().endsWith(".gguf")) {
      budget.n--
      out.push(full)
    }
  }
}

export async function scanModels(): Promise<ModelEntry[]> {
  const budget = { n: 4000 }
  const seen = new Set<string>()
  const entries: ModelEntry[] = []

  for (const { dir, label } of searchRoots()) {
    const files: string[] = []
    await walk(dir, 4, files, budget)
    for (const path of files) {
      if (seen.has(path)) continue
      seen.add(path)
      const name = basename(path)
      const shard = SHARD_RE.exec(name)
      // Only surface the first shard of a split model; the loader follows the rest.
      if (shard && shard[2] !== "00001") continue
      let size = 0
      let mtime = 0
      try {
        const st = await stat(path)
        size = st.size
        mtime = st.mtimeMs
      } catch {
        continue
      }
      entries.push({
        path,
        name: shard ? `${shard[1]}.gguf` : name,
        size,
        mtime,
        shards: shard ? Number(shard[3]) : 1,
        source: label,
      })
    }
  }

  entries.sort((a, b) => b.mtime - a.mtime)
  return entries
}

/** Read headers with bounded concurrency, reporting each result as it lands. */
export async function enrich(
  entries: ModelEntry[],
  onOne: (path: string, facts?: ModelFacts, error?: string) => void,
  concurrency = 4,
) {
  let i = 0
  const workers = Array.from({ length: Math.min(concurrency, entries.length) }, async () => {
    for (;;) {
      const idx = i++
      if (idx >= entries.length) return
      const e = entries[idx]!
      try {
        const header = await readGgufHeader(e.path)
        onOne(e.path, factsFrom(header, e.name.replace(/\.gguf$/i, "")))
      } catch (err) {
        onOne(e.path, undefined, err instanceof Error ? err.message : String(err))
      }
    }
  })
  await Promise.all(workers)
}
