import { describe, expect, test } from "bun:test"

import { buildArgs, type ServeOptions } from "./server.js"
import type { Launcher } from "./binary.js"

const launcher: Launcher = {
  cmd: "/workspace/oxidize-c/oxidize-c",
  prefix: [],
  label: "test",
  cwd: "/workspace",
}

function serve(partial: Partial<ServeOptions> = {}): ServeOptions {
  return { model: "/models/x.gguf", ...partial }
}

describe("C serve argv", () => {
  test("positional model plus host and port", () => {
    expect(buildArgs(launcher, serve(), "127.0.0.1", 8080)).toEqual([
      "serve",
      "/models/x.gguf",
      "--host",
      "127.0.0.1",
      "--port",
      "8080",
    ])
  })

  test("forwards backend, threads, ctx-size, and max-tokens", () => {
    expect(
      buildArgs(
        launcher,
        serve({ backend: "cpu", threads: 8, ctxSize: 4096, maxTokens: 128 }),
        "0.0.0.0",
        9,
      ),
    ).toEqual([
      "serve",
      "/models/x.gguf",
      "--host",
      "0.0.0.0",
      "--port",
      "9",
      "--backend",
      "cpu",
      "--threads",
      "8",
      "--ctx-size",
      "4096",
      "--max-tokens",
      "128",
    ])
  })

  test("skips auto backend", () => {
    const args = buildArgs(launcher, serve({ backend: "auto" }), "127.0.0.1", 1)
    expect(args).not.toContain("--backend")
  })
})
