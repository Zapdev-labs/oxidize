import { describe, expect, test } from "bun:test"

import { streamChat } from "./api.js"

const sse = (chunks: string[]) =>
  chunks.map((c) => `data: ${c}\n\n`).join("") + "data: [DONE]\n\n"

describe("streamChat", () => {
  test("assembles SSE deltas and reports first token", async () => {
    const body = sse([
      JSON.stringify({ choices: [{ delta: { role: "assistant" } }] }),
      JSON.stringify({ choices: [{ delta: { content: "Hel" } }] }),
      JSON.stringify({ choices: [{ delta: { content: "lo" } }] }),
      "{not json}",
      JSON.stringify({ choices: [{ delta: {} }], usage: { prompt_tokens: 7, completion_tokens: 2 } }),
    ])
    const server = Bun.serve({
      port: 0,
      fetch: () => new Response(body, { headers: { "content-type": "text/event-stream" } }),
    })

    let out = ""
    let first = 0
    const usage: { prompt: number; completion: number }[] = []
    try {
      await streamChat(
        `http://127.0.0.1:${server.port}`,
        "m",
        [{ role: "user", content: "hi" }],
        { temperature: 0.1, topP: 1, maxTokens: 16 },
        {
          onFirstToken: () => first++,
          onDelta: (d) => (out += d),
          onUsage: (u) => usage.push(u),
        },
        new AbortController().signal,
      )
    } finally {
      server.stop(true)
    }

    expect(out).toBe("Hello")
    expect(first).toBe(1)
    expect(usage).toEqual([{ prompt: 7, completion: 2 }])
  })

  test("surfaces server errors with the response body", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => new Response("model not loaded", { status: 500 }),
    })
    try {
      await expect(
        streamChat(
          `http://127.0.0.1:${server.port}`,
          "m",
          [{ role: "user", content: "hi" }],
          { temperature: 0.1, topP: 1, maxTokens: 16 },
          { onDelta: () => {} },
          new AbortController().signal,
        ),
      ).rejects.toThrow(/500 model not loaded/)
    } finally {
      server.stop(true)
    }
  })
})
