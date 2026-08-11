import { c, glyph } from "../theme.js"
import { useStore, type ServerStatus, type View } from "../state/store.js"
import { ellipsizeStart, num } from "../util/format.js"

const STATUS_COLOR: Record<ServerStatus, string> = {
  idle: c.faint,
  starting: c.warn,
  loading: c.warn,
  ready: c.ok,
  error: c.err,
  stopped: c.faint,
}

const VIEWS: { id: View; label: string }[] = [
  { id: "chat", label: "chat" },
  { id: "models", label: "models" },
  { id: "monitor", label: "monitor" },
  { id: "logs", label: "logs" },
]

export function Header({ width, spin }: { width: number; spin: string }) {
  const server = useStore((s) => s.server)
  const tps = useStore((s) => s.chat.tps)
  const generating = useStore((s) => s.chat.generating)

  const busy = server.status === "starting" || server.status === "loading"
  const label =
    server.status === "ready"
      ? server.external
        ? "attached"
        : "ready"
      : busy
        ? server.detail || server.status
        : server.status === "error"
          ? `error: ${server.detail}`
          : "no model"

  const model = server.modelPath
    ? ellipsizeStart(server.modelId, Math.max(16, Math.floor(width * 0.35)))
    : "—"

  return (
    <box
      style={{
        flexDirection: "row",
        justifyContent: "space-between",
        backgroundColor: c.panelAlt,
        paddingLeft: 1,
        paddingRight: 1,
        height: 1,
        flexShrink: 0,
      }}
    >
      <text wrapMode="none" truncate>
        <strong fg={c.accent}>OXIDIZE</strong>
        <span fg={c.faint}>{"  " + glyph.bar + " "}</span>
        <span fg={c.text}>{model}</span>
        {server.status === "ready" ? (
          <span fg={c.faint}>{`  ${server.backend}${server.threads ? `/${server.threads}t` : ""}`}</span>
        ) : null}
      </text>
      <text>
        {generating || tps > 0 ? (
          <span fg={c.accent}>{`${num(tps, 1)} tok/s  `}</span>
        ) : null}
        <span fg={STATUS_COLOR[server.status]}>{busy ? spin : glyph.dot}</span>
        <span fg={c.dim}>{` ${label}`}</span>
      </text>
    </box>
  )
}

export function Tabs({ active }: { active: View }) {
  return (
    <box
      style={{
        flexDirection: "row",
        backgroundColor: c.bg,
        height: 1,
        flexShrink: 0,
        paddingLeft: 1,
      }}
    >
      {VIEWS.map((v, i) => {
        const on = v.id === active
        return (
          <text key={v.id}>
            <span fg={on ? c.bg : c.faint} bg={on ? c.accent : undefined}>
              {` ${i + 1} ${v.label} `}
            </span>
            <span> </span>
          </text>
        )
      })}
    </box>
  )
}

export function StatusBar({ hints, width }: { hints: [string, string][]; width: number }) {
  const toast = useStore((s) => s.toast)
  const fresh = toast && Date.now() - toast.at < 6000
  const tone = toast?.tone === "err" ? c.err : toast?.tone === "ok" ? c.ok : c.steel

  // Drop hints right-to-left until they clear the right-hand slot; each renders as
  // " key " + " desc   " = key + desc + 6 columns.
  const right = fresh ? toast!.text.length : "ctrl+k commands".length
  const cost = (h: [string, string]) => h[0].length + h[1].length + 6
  let shown = hints
  while (shown.length > 1 && shown.reduce((a, h) => a + cost(h), 0) + right + 3 > width) {
    shown = shown.slice(0, -1)
  }

  return (
    <box
      style={{
        flexDirection: "row",
        justifyContent: "space-between",
        backgroundColor: c.panelAlt,
        height: 1,
        flexShrink: 0,
        paddingLeft: 1,
        paddingRight: 1,
      }}
    >
      <text wrapMode="none" truncate>
        {shown.map(([k, d], i) => (
          <span key={i}>
            <span fg={c.dim} bg={c.raised}>{` ${k} `}</span>
            <span fg={c.faint}>{` ${d}   `}</span>
          </span>
        ))}
      </text>
      {fresh ? (
        <text fg={tone} wrapMode="none" truncate>
          {toast!.text}
        </text>
      ) : (
        <text fg={c.faint} wrapMode="none" truncate>
          ctrl+k commands
        </text>
      )}
    </box>
  )
}
