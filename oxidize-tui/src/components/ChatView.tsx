import type { InputRenderable } from "@opentui/core"
import { useEffect, useRef } from "react"

import { send } from "../state/actions.js"
import { useStore, type Message } from "../state/store.js"
import { c, glyph, roleColor } from "../theme.js"
import { clock, ms, num, params as fmtParams } from "../util/format.js"
import { Markdown } from "./Markdown.js"
import { Field, Sparkline } from "./primitives.js"

function Bubble({ msg, spin }: { msg: Message; spin: string }) {
  const color = msg.error ? c.err : roleColor(msg.role)
  const who = msg.role === "user" ? "you" : msg.role === "assistant" ? "oxidize" : "system"
  const s = msg.stats

  return (
    <box style={{ flexDirection: "column", width: "100%", marginBottom: 1 }}>
      <box style={{ flexDirection: "row", justifyContent: "space-between", width: "100%" }}>
        <text>
          <span fg={color}>{glyph.bar} </span>
          <strong fg={color}>{who}</strong>
          {msg.streaming ? <span fg={c.accent}>{` ${spin}`}</span> : null}
        </text>
        <text fg={c.faint}>
          {s && (s.tokens > 0 || s.ttft != null)
            ? `${s.tokens} tok · ttft ${ms(s.ttft)} · ${num(s.elapsed && s.ttft != null ? (s.tokens / Math.max(1, s.elapsed - s.ttft)) * 1000 : 0)} tok/s`
            : clock(msg.at)}
        </text>
      </box>
      <box style={{ paddingLeft: 2, width: "100%" }}>
        {msg.role === "user" ? (
          <text fg={c.text} wrapMode="word">
            {msg.content}
          </text>
        ) : msg.content ? (
          <Markdown content={msg.content} dim={msg.error} />
        ) : (
          <text fg={c.faint}>thinking…</text>
        )}
      </box>
    </box>
  )
}

function Empty() {
  const status = useStore((s) => s.server.status)
  return (
    <box style={{ flexDirection: "column", paddingTop: 1, paddingLeft: 1 }}>
      <ascii-font text="oxidize" font="tiny" color={c.accentDim} />
      <box style={{ height: 1 }} />
      <text fg={c.dim}>
        {status === "ready"
          ? "Model is up. Ask it something."
          : "No model loaded yet."}
      </text>
      <box style={{ height: 1 }} />
      <text fg={c.faint}>
        <span fg={c.dim}>ctrl+t</span> cycle views · <span fg={c.dim}>2</span> model browser ·{" "}
        <span fg={c.dim}>ctrl+k</span> commands · <span fg={c.dim}>?</span> help
      </text>
    </box>
  )
}

function Sidebar() {
  const server = useStore((s) => s.server)
  const p = useStore((s) => s.params)
  const tpsHistory = useStore((s) => s.chat.tpsHistory)
  const tps = useStore((s) => s.chat.tps)
  const entry = useStore((s) => s.models.entries.find((e) => e.path === s.server.modelPath))
  const facts = entry?.facts

  return (
    <box
      style={{
        width: 30,
        flexShrink: 0,
        flexDirection: "column",
        backgroundColor: c.panel,
        border: ["left"],
        borderColor: c.border,
        paddingLeft: 1,
        paddingRight: 1,
      }}
    >
      <text fg={c.faint}>RUNTIME</text>
      <Field k="backend" v={server.backend} />
      <Field k="endpoint" v={server.url?.replace("http://", "") ?? "—"} />
      <Field k="pid" v={server.pid ? String(server.pid) : server.external ? "external" : "—"} />
      <box style={{ height: 1 }} />

      <text fg={c.faint}>MODEL</text>
      <Field k="arch" v={facts?.arch ?? "—"} />
      <Field k="quant" v={facts?.quant ?? "—"} color={facts ? c.steel : c.text} />
      <Field k="params" v={facts ? fmtParams(facts.params) : "—"} />
      <Field k="layers" v={facts?.layers ? String(facts.layers) : "—"} />
      <Field k="ctx" v={facts?.ctx ? facts.ctx.toLocaleString() : "—"} />
      {facts?.expertCount ? <Field k="experts" v={String(facts.expertCount)} /> : null}
      <box style={{ height: 1 }} />

      <text fg={c.faint}>SAMPLING</text>
      <Field k="temp" v={num(p.temperature, 2)} />
      <Field k="top_p" v={num(p.topP, 2)} />
      <Field k="top_k" v={p.topK ? String(p.topK) : "off"} />
      <Field k="max_tok" v={String(p.maxTokens)} />
      <box style={{ height: 1 }} />

      <text fg={c.faint}>THROUGHPUT</text>
      <text>
        <span fg={c.accent}>{num(tps, 1)}</span>
        <span fg={c.faint}> tok/s</span>
      </text>
      <Sparkline data={tpsHistory} width={26} />
    </box>
  )
}

export function ChatView({ focused, width, spin }: { focused: boolean; width: number; spin: string }) {
  const messages = useStore((s) => s.chat.messages)
  const generating = useStore((s) => s.chat.generating)
  const ready = useStore((s) => s.server.status === "ready")
  const inputRef = useRef<InputRenderable>(null)
  const wide = width >= 96

  useEffect(() => {
    if (focused) inputRef.current?.focus?.()
  }, [focused])

  return (
    <box style={{ flexDirection: "row", flexGrow: 1 }}>
      <box style={{ flexDirection: "column", flexGrow: 1 }}>
        <scrollbox
          style={{
            flexGrow: 1,
            rootOptions: { backgroundColor: c.bg },
            viewportOptions: { backgroundColor: c.bg },
            contentOptions: { backgroundColor: c.bg, paddingLeft: 1, paddingRight: 1, paddingTop: 1 },
            scrollbarOptions: {
              showArrows: false,
              trackOptions: { foregroundColor: c.borderHot, backgroundColor: c.panel },
            },
          }}
          stickyScroll
          stickyStart="bottom"
        >
          {messages.length === 0 ? (
            <Empty />
          ) : (
            messages.map((m) => <Bubble key={m.id} msg={m} spin={spin} />)
          )}
        </scrollbox>

        <box
          style={{
            height: 3,
            flexShrink: 0,
            border: true,
            borderColor: focused ? c.accentDim : c.border,
            backgroundColor: c.panel,
            paddingLeft: 1,
            paddingRight: 1,
          }}
          title={generating ? " generating — esc to cancel " : ready ? " message " : " no model "}
          titleColor={generating ? c.accent : c.dim}
        >
          <input
            ref={inputRef}
            focused={focused}
            placeholder={ready ? "Ask oxidize…" : "Load a model first (press 2)"}
            textColor={c.text}
            placeholderColor={c.faint}
            cursorColor={c.accent}
            backgroundColor={c.panel}
            onSubmit={(submitted) => {
              const value = typeof submitted === "string" ? submitted : (inputRef.current?.value ?? "")
              if (!value.trim()) return
              if (inputRef.current) inputRef.current.value = ""
              void send(value)
            }}
          />
        </box>
      </box>
      {wide ? <Sidebar /> : null}
    </box>
  )
}
