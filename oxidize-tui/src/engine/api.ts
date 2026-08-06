/** OpenAI-compatible client for the oxidize server. */

export interface ChatMessage {
  role: "system" | "user" | "assistant"
  content: string
}

export interface GenParams {
  temperature: number
  topP: number
  topK?: number
  maxTokens: number
}

export interface StreamCallbacks {
  onFirstToken?: () => void
  onDelta: (text: string) => void
  onUsage?: (usage: { prompt: number; completion: number }) => void
}

export async function listModelIds(url: string): Promise<string[]> {
  const r = await fetch(`${url}/v1/models`, { signal: AbortSignal.timeout(3000) })
  if (!r.ok) throw new Error(`GET /v1/models -> ${r.status}`)
  const body = (await r.json()) as { data?: { id: string }[] }
  return (body.data ?? []).map((m) => m.id)
}

export async function streamChat(
  url: string,
  model: string,
  messages: ChatMessage[],
  params: GenParams,
  cb: StreamCallbacks,
  signal: AbortSignal,
): Promise<void> {
  const body: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    temperature: params.temperature,
    top_p: params.topP,
    max_tokens: params.maxTokens,
  }
  if (params.topK && params.topK > 0) body.top_k = params.topK

  const res = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  })

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "")
    throw new Error(`chat/completions -> ${res.status} ${detail.slice(0, 400)}`)
  }

  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ""
  let first = true

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })

    let sep: number
    while ((sep = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, sep).replace(/\r$/, "")
      buf = buf.slice(sep + 1)
      if (!line.startsWith("data:")) continue
      const payload = line.slice(5).trim()
      if (!payload || payload === "[DONE]") continue

      let chunk: any
      try {
        chunk = JSON.parse(payload)
      } catch {
        continue
      }
      const delta: string | undefined =
        chunk?.choices?.[0]?.delta?.content ?? chunk?.choices?.[0]?.text
      if (delta) {
        if (first) {
          first = false
          cb.onFirstToken?.()
        }
        cb.onDelta(delta)
      }
      if (chunk?.usage) {
        cb.onUsage?.({
          prompt: chunk.usage.prompt_tokens ?? 0,
          completion: chunk.usage.completion_tokens ?? 0,
        })
      }
    }
  }
}
