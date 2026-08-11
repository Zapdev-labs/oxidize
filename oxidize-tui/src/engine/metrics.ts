/** Prometheus text scrape -> the handful of series the monitor renders. */

export interface MetricsSnapshot {
  at: number
  tokensPerSecond: number
  tokensGenerated: number
  tokensPrompt: number
  requestsInFlight: number
  requestsTotal: number
  queueDepth: number
  cacheHitRatio: number
  kvCacheBytes: number
  kvBlocksUsed: number
  kvBlocksTotal: number
  errorsTotal: number
  realtimeConnections: number
  /** mean seconds per inference, from the histogram sum/count pair */
  inferenceMean: number
}

function parseFlat(text: string): Map<string, number> {
  const out = new Map<string, number>()
  for (const raw of text.split("\n")) {
    const line = raw.trim()
    if (!line || line.startsWith("#")) continue
    const sp = line.lastIndexOf(" ")
    if (sp < 0) continue
    const key = line.slice(0, sp)
    const value = Number(line.slice(sp + 1))
    if (!Number.isFinite(value)) continue
    // Sum across label sets: `foo{a="b"} 1` and `foo{a="c"} 2` -> foo = 3
    const brace = key.indexOf("{")
    const name = brace < 0 ? key : key.slice(0, brace)
    if (name === "oxidize_requests_total" && /(?:^|,)path="\/metrics"(?:,|})/.test(key.slice(brace))) {
      continue
    }
    out.set(name, (out.get(name) ?? 0) + value)
  }
  return out
}

export async function scrape(
  url: string,
  timeoutMs = 2000,
  signal?: AbortSignal,
): Promise<MetricsSnapshot> {
  const timeout = AbortSignal.timeout(timeoutMs)
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout
  const res = await fetch(`${url}/metrics`, { signal: combined })
  if (!res.ok) throw new Error(`GET /metrics -> ${res.status}`)
  const m = parseFlat(await res.text())
  const g = (k: string) => m.get(k) ?? 0
  const sum = g("oxidize_model_inference_duration_seconds_sum")
  const count = g("oxidize_model_inference_duration_seconds_count")

  return {
    at: Date.now(),
    tokensPerSecond: g("oxidize_tokens_per_second"),
    tokensGenerated: g("oxidize_tokens_generated_total"),
    tokensPrompt: g("oxidize_tokens_prompt_total"),
    requestsInFlight: g("oxidize_requests_in_flight"),
    requestsTotal: g("oxidize_requests_total"),
    queueDepth: g("oxidize_queue_depth"),
    cacheHitRatio: g("oxidize_cache_hit_ratio"),
    kvCacheBytes: g("oxidize_kv_cache_size_bytes"),
    kvBlocksUsed: g("oxidize_kv_cache_blocks_used"),
    kvBlocksTotal: g("oxidize_kv_cache_blocks_total"),
    errorsTotal: g("oxidize_errors_total"),
    realtimeConnections: g("oxidize_realtime_connections"),
    inferenceMean: count > 0 ? sum / count : 0,
  }
}
