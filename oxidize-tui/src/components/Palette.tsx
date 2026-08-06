import type { InputRenderable } from "@opentui/core"
import { basename } from "node:path"
import { useEffect, useMemo, useRef } from "react"

import {
  cancelGeneration,
  clearChat,
  loadModel,
  rescanModels,
  restartServer,
  stopServer,
} from "../state/actions.js"
import { getState, set, setIn, toast, type State } from "../state/store.js"
import { c } from "../theme.js"
import { fuzzy, pad } from "../util/format.js"

export interface Command {
  id: string
  title: string
  hint: string
  group: string
  run: () => void
}

const bump = (key: "temperature" | "topP", delta: number) => () => {
  const cur = getState().params[key]
  const next = Math.min(2, Math.max(0, Number((cur + delta).toFixed(2))))
  setIn("params", { [key]: next } as never)
  toast(`${key} = ${next}`)
}

export function commands(s: State): Command[] {
  const list: Command[] = [
    { id: "view.chat", title: "Go to chat", hint: "1", group: "view", run: () => set({ view: "chat" }) },
    { id: "view.models", title: "Go to models", hint: "2", group: "view", run: () => set({ view: "models" }) },
    { id: "view.monitor", title: "Go to monitor", hint: "3", group: "view", run: () => set({ view: "monitor" }) },
    { id: "view.logs", title: "Go to logs", hint: "4", group: "view", run: () => set({ view: "logs" }) },

    { id: "chat.clear", title: "Clear conversation", hint: "", group: "chat", run: clearChat },
    {
      id: "chat.cancel",
      title: "Cancel generation",
      hint: "esc",
      group: "chat",
      run: cancelGeneration,
    },
    {
      id: "chat.copy",
      title: "Copy last reply to clipboard",
      hint: "",
      group: "chat",
      run: () => {
        const last = [...s.chat.messages].reverse().find((m) => m.role === "assistant")
        if (!last) return toast("nothing to copy", "err")
        Bun.spawn(["sh", "-c", "command -v wl-copy >/dev/null && wl-copy || xclip -selection clipboard"], {
          stdin: new TextEncoder().encode(last.content),
        })
        toast("copied")
      },
    },

    { id: "srv.rescan", title: "Rescan model directories", hint: "r", group: "server", run: () => void rescanModels() },
    { id: "srv.restart", title: "Restart server", hint: "", group: "server", run: restartServer },
    { id: "srv.stop", title: "Stop server", hint: "", group: "server", run: () => stopServer() },

    { id: "p.temp+", title: "Temperature +0.1", hint: `${s.params.temperature}`, group: "sampling", run: bump("temperature", 0.1) },
    { id: "p.temp-", title: "Temperature -0.1", hint: `${s.params.temperature}`, group: "sampling", run: bump("temperature", -0.1) },
    { id: "p.topp+", title: "top_p +0.05", hint: `${s.params.topP}`, group: "sampling", run: bump("topP", 0.05) },
    { id: "p.topp-", title: "top_p -0.05", hint: `${s.params.topP}`, group: "sampling", run: bump("topP", -0.05) },
    {
      id: "p.maxtok",
      title: "Cycle max tokens (256/512/1024/2048/4096)",
      hint: `${s.params.maxTokens}`,
      group: "sampling",
      run: () => {
        const steps = [256, 512, 1024, 2048, 4096]
        const idx = steps.indexOf(getState().params.maxTokens)
        const next = steps[(idx + 1) % steps.length]!
        setIn("params", { maxTokens: next })
        toast(`max_tokens = ${next}`)
      },
    },
    {
      id: "srv.backend",
      title: "Cycle backend (applies on next load)",
      hint: s.server.backend,
      group: "server",
      run: () => {
        const backends = ["cpu", "cuda", "metal", "vulkan", "rocm", "mlx"]
        const idx = backends.indexOf(getState().server.backend)
        const next = backends[(idx + 1) % backends.length]!
        setIn("server", { backend: next })
        toast(`backend = ${next} (reload the model to apply)`)
      },
    },
    { id: "logs.follow", title: "Toggle log follow", hint: "f", group: "logs", run: () => setIn("logs", (l) => ({ follow: !l.follow })) },
    { id: "logs.clear", title: "Clear logs", hint: "", group: "logs", run: () => setIn("logs", { lines: [] }) },
    { id: "app.help", title: "Keyboard help", hint: "?", group: "app", run: () => set({ overlay: { kind: "help" } }) },
    { id: "app.quit", title: "Quit", hint: "ctrl+c", group: "app", run: () => process.exit(0) },
  ]

  for (const e of s.models.entries.slice(0, 200)) {
    list.push({
      id: `load:${e.path}`,
      title: `Load ${e.name}`,
      hint: `${e.facts?.quant ?? ""} ${e.source}`.trim(),
      group: "model",
      run: () => void loadModel(e.path),
    })
  }
  return list
}

export function Palette({ query, cursor, width }: { query: string; cursor: number; width: number }) {
  const inputRef = useRef<InputRenderable>(null)
  const matches = useMemo(() => paletteMatches(query), [query])

  useEffect(() => {
    inputRef.current?.focus?.()
  }, [])

  const boxWidth = Math.min(76, Math.max(40, width - 8))
  const sel = matches.length ? Math.min(cursor, matches.length - 1) : 0

  return (
    <box
      style={{
        position: "absolute",
        left: Math.max(2, Math.floor((width - boxWidth) / 2)),
        top: 3,
        width: boxWidth,
        zIndex: 100,
        flexDirection: "column",
        backgroundColor: c.panelAlt,
        border: true,
        borderColor: c.accentDim,
        paddingLeft: 1,
        paddingRight: 1,
      }}
      title=" command "
      titleColor={c.accent}
    >
      <box style={{ height: 1, flexDirection: "row" }}>
        <text fg={c.accent}>{"› "}</text>
        <input
          ref={inputRef}
          focused
          placeholder="type a command…"
          textColor={c.text}
          placeholderColor={c.faint}
          cursorColor={c.accent}
          backgroundColor={c.panelAlt}
          onInput={(v) => set({ overlay: { kind: "palette", query: v, cursor: 0 } })}
        />
      </box>
      <box style={{ height: 1 }} />
      {matches.length === 0 ? (
        <text fg={c.faint}>no matches</text>
      ) : (
        matches.map((cmd, i) => (
          <text key={cmd.id} bg={i === sel ? c.raised : undefined}>
            <span fg={i === sel ? c.accent : c.faint}>{i === sel ? "▸ " : "  "}</span>
            <span fg={c.faint}>{pad(cmd.group, 9)}</span>
            <span fg={i === sel ? c.text : c.dim}>{pad(cmd.title, boxWidth - 24)}</span>
            <span fg={c.faint}>{cmd.hint}</span>
          </text>
        ))
      )}
    </box>
  )
}

export function paletteMatches(query: string): Command[] {
  const all = commands(getState())
  if (!query.trim()) return all.slice(0, 12)
  return all
    .map((cmd) => ({ cmd, score: fuzzy(query, `${cmd.group} ${cmd.title}`) }))
    .filter((x) => x.score != null)
    .sort((a, b) => a.score! - b.score!)
    .slice(0, 12)
    .map((x) => x.cmd)
}

export function Help({ width }: { width: number }) {
  const rows: [string, string][] = [
    ["ctrl+t", "cycle view"],
    ["1 2 3 4", "jump to view (outside chat input)"],
    ["ctrl+k", "command palette"],
    ["?", "this help (outside chat input)"],
    ["esc", "cancel generation / close overlay"],
    ["ctrl+c", "quit"],
    ["", ""],
    ["chat: ⏎", "send"],
    ["chat: ctrl+l", "clear conversation"],
    ["", ""],
    ["models: j/k ↑/↓", "move"],
    ["models: ⏎", "load model"],
    ["models: i", "inspect GGUF header"],
    ["models: /", "filter"],
    ["models: r", "rescan"],
    ["", ""],
    ["logs: f", "toggle follow"],
    ["logs: c", "clear"],
  ]
  const boxWidth = Math.min(60, Math.max(38, width - 10))
  return (
    <box
      style={{
        position: "absolute",
        left: Math.max(2, Math.floor((width - boxWidth) / 2)),
        top: 3,
        width: boxWidth,
        zIndex: 100,
        flexDirection: "column",
        backgroundColor: c.panelAlt,
        border: true,
        borderColor: c.borderHot,
        paddingLeft: 2,
        paddingRight: 2,
      }}
      title=" keys "
      titleColor={c.accent}
      bottomTitle=" esc "
      bottomTitleAlignment="right"
    >
      {rows.map(([k, v], i) =>
        k === "" ? (
          <box key={i} style={{ height: 1 }} />
        ) : (
          <text key={i}>
            <span fg={c.accent}>{pad(k, 16)}</span>
            <span fg={c.dim}>{v}</span>
          </text>
        ),
      )}
    </box>
  )
}

export const displayName = (p: string) => basename(p)
