import { describe, expect, test } from "bun:test"

import { scrape } from "./metrics.js"

const SAMPLE = `# HELP oxidize_tokens_per_second Current token generation throughput
# TYPE oxidize_tokens_per_second gauge
oxidize_tokens_per_second 42.5
oxidize_tokens_generated_total 1200
oxidize_requests_total{method="POST",path="/v1/chat/completions",status="200"} 3
oxidize_requests_total{method="GET",path="/v1/models",status="200"} 4
oxidize_requests_total{method="GET",path="/metrics",status="200"} 99
oxidize_kv_cache_blocks_used 12
oxidize_kv_cache_blocks_total 64
oxidize_model_inference_duration_seconds_sum 2.5
oxidize_model_inference_duration_seconds_count 5
`

describe("metrics scrape", () => {
  test("parses gauges, sums label sets, and derives the histogram mean", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => new Response(SAMPLE, { headers: { "content-type": "text/plain" } }),
    })
    try {
      const snap = await scrape(`http://127.0.0.1:${server.port}`)
      expect(snap.tokensPerSecond).toBe(42.5)
      expect(snap.tokensGenerated).toBe(1200)
      expect(snap.requestsTotal).toBe(7)
      expect(snap.kvBlocksUsed).toBe(12)
      expect(snap.kvBlocksTotal).toBe(64)
      expect(snap.inferenceMean).toBeCloseTo(0.5)
    } finally {
      server.stop(true)
    }
  })

  test("throws on a non-200", async () => {
    const server = Bun.serve({ port: 0, fetch: () => new Response("nope", { status: 503 }) })
    try {
      await expect(scrape(`http://127.0.0.1:${server.port}`)).rejects.toThrow(/503/)
    } finally {
      server.stop(true)
    }
  })
})
