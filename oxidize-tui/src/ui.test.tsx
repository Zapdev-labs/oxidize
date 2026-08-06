import { testRender } from "@opentui/react/test-utils"
import { afterEach, describe, expect, test } from "bun:test"

import { App } from "./App.js"
import { nextId, set, setIn } from "./state/store.js"

function seed() {
  set({ view: "chat", overlay: null, toast: null })
  setIn("server", {
    status: "ready",
    url: "http://127.0.0.1:41234",
    pid: 4242,
    modelPath: "/models/Qwen3-4B-Q4_K_M.gguf",
    modelId: "Qwen3-4B-Q4_K_M",
    backend: "cpu",
    threads: 16,
    external: false,
    startedAt: Date.now() - 90_000,
    detail: "",
  })
  setIn("models", {
    entries: [
      {
        path: "/models/Qwen3-4B-Q4_K_M.gguf",
        name: "Qwen3-4B-Q4_K_M.gguf",
        size: 2_600_000_000,
        mtime: Date.now(),
        shards: 1,
        source: "workspace",
        facts: {
          arch: "qwen3",
          quant: "Q4_K",
          params: 4_020_000_000,
          ctx: 40960,
          layers: 36,
          embed: 2560,
          heads: 32,
          expertCount: 0,
          name: "Qwen3 4B",
          tensorCount: 398,
          version: 3,
        },
      },
      {
        path: "/models/LongCat-Flash-AL5.gguf",
        name: "LongCat-Flash-AL5.gguf",
        size: 320_000_000_000,
        mtime: Date.now() - 1000,
        shards: 7,
        source: "hf-cache",
        facts: {
          arch: "longcat",
          quant: "AL5",
          params: 560_000_000_000,
          ctx: 131072,
          layers: 56,
          embed: 6144,
          heads: 64,
          expertCount: 512,
          name: "LongCat Flash",
          tensorCount: 1204,
          version: 3,
        },
      },
    ],
    scanning: false,
    cursor: 0,
    filter: "",
    filtering: false,
    inspect: null,
  })
  setIn("chat", {
    messages: [
      { id: nextId(), role: "user", content: "What quant should I use on a dual-socket Xeon?", at: Date.now() },
      {
        id: nextId(),
        role: "assistant",
        content:
          "Use **Q4_K_M** for dense models under 70B.\n\n- fits in L3 better than Q5\n- `--numa single --threads 16` on your box\n\n```bash\noxidize run model.gguf --numa single\n```",
        at: Date.now(),
        stats: { ttft: 180, tokens: 64, elapsed: 1600 },
      },
    ],
    generating: false,
    tps: 41.7,
    tpsHistory: [10, 22, 31, 38, 41, 44, 42, 41.7],
  })
}

let teardown: (() => void) | null = null
afterEach(() => {
  teardown?.()
  teardown = null
})

async function render(width = 120, height = 32) {
  seed()
  const t = await testRender(<App />, { width, height })
  teardown = () => t.renderer.destroy()
  await t.flush()
  return t
}

describe("app frames", () => {
  test("chat view shows chrome, transcript and sidebar", async () => {
    const t = await render()
    const frame = t.captureCharFrame()

    expect(frame).toContain("OXIDIZE")
    expect(frame).toContain("Qwen3-4B-Q4_K_M")
    expect(frame).toContain("ready")
    expect(frame).toContain("1 chat")
    expect(frame).toContain("4 logs")
    expect(frame).toContain("What quant should I use")
    expect(frame).toContain("oxidize run model.gguf") // fenced code block
    expect(frame).toContain("THROUGHPUT") // sidebar at 120 cols
    expect(frame).toContain("41.7")
  })

  test("narrow terminals drop the sidebar instead of clipping the transcript", async () => {
    const t = await render(72, 24)
    const frame = t.captureCharFrame()
    expect(frame).not.toContain("THROUGHPUT")
    expect(frame).toContain("What quant")
  })

  test("model browser lists rows with quant and parameter columns", async () => {
    seed()
    set({ view: "models" })
    const t = await testRender(<App />, { width: 120, height: 32 })
    teardown = () => t.renderer.destroy()
    await t.flush()
    const frame = t.captureCharFrame()

    expect(frame).toContain("NAME")
    expect(frame).toContain("QUANT")
    expect(frame).toContain("Qwen3-4B-Q4_K_M.gguf")
    expect(frame).toContain("AL5")
    expect(frame).toContain("7 shards")
    expect(frame).toContain("2/2 models")
  })

  test("model rows fit the terminal instead of wrapping", async () => {
    seed()
    set({ view: "models" })
    const t = await testRender(<App />, { width: 120, height: 32 })
    teardown = () => t.renderer.destroy()
    await t.flush()

    const lines = t.captureCharFrame().split("\n")
    for (const line of lines) expect(line.length).toBeLessThanOrEqual(120)
    // The two model rows are adjacent — nothing wrapped between them.
    const first = lines.findIndex((l) => l.includes("Qwen3-4B-Q4_K_M.gguf"))
    const second = lines.findIndex((l) => l.includes("LongCat-Flash-AL5.gguf"))
    expect(second).toBe(first + 1)
    expect(lines[second]).toContain("hf-cache")
  })

  test("narrow model browser drops columns right-to-left", async () => {
    seed()
    set({ view: "models" })
    const t = await testRender(<App />, { width: 76, height: 20 })
    teardown = () => t.renderer.destroy()
    await t.flush()
    const frame = t.captureCharFrame()

    expect(frame).toContain("QUANT") // high-priority columns survive
    expect(frame).toContain("PARAMS")
    expect(frame).not.toContain("SOURCE") // low-priority ones are dropped
    expect(frame).not.toContain("CTX")
    // Status-bar hints shrink rather than colliding with the right-hand slot.
    expect(frame).toContain("ctrl+k commands")
    expect(frame).not.toContain("rescanctrl+k")
  })

  test("ctrl+k opens the command palette and filters", async () => {
    const t = await render()
    t.mockInput.pressKey("k", { ctrl: true })
    await t.flush()
    expect(t.captureCharFrame()).toContain("command")

    t.mockInput.typeText("monitor")
    await t.flush()
    const frame = t.captureCharFrame()
    expect(frame).toContain("Go to monitor")
    expect(frame).not.toContain("Clear conversation")
  })

  test("number keys switch views outside the chat input", async () => {
    seed()
    set({ view: "models" })
    const t = await testRender(<App />, { width: 120, height: 32 })
    teardown = () => t.renderer.destroy()
    await t.flush()

    t.mockInput.pressKey("3")
    await t.flush()
    expect(t.captureCharFrame()).toContain("throughput")
  })
})
