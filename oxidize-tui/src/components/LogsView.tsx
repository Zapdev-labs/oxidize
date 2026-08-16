import { useStore } from "../state/store.js"
import { c } from "../theme.js"
import { clock } from "../util/format.js"

function lineColor(stream: string, text: string): string {
  if (stream === "tui") return c.steel
  if (/\bERROR\b|\berror\b|panicked/.test(text)) return c.err
  if (/\bWARN\b|\bwarning\b/.test(text)) return c.warn
  if (stream === "err") return c.dim
  return c.text
}

export function LogsView({ focused }: { focused: boolean }) {
  const { lines, follow } = useStore((s) => s.logs)

  return (
    <box style={{ flexGrow: 1, flexDirection: "column", backgroundColor: c.bg }}>
      <scrollbox
        focused={focused}
        stickyScroll={follow}
        stickyStart="bottom"
        style={{
          flexGrow: 1,
          rootOptions: { backgroundColor: c.bg },
          viewportOptions: { backgroundColor: c.bg },
          contentOptions: { backgroundColor: c.bg, paddingLeft: 1, paddingRight: 1 },
          scrollbarOptions: {
            showArrows: false,
            trackOptions: { foregroundColor: c.borderHot, backgroundColor: c.panel },
          },
        }}
      >
        {lines.length === 0 ? (
          <text fg={c.faint}>No server output yet.</text>
        ) : (
          lines.map((l) => (
            <text key={l.id} wrapMode="word">
              <span fg={c.faint}>{clock(l.at) + " "}</span>
              <span fg={lineColor(l.stream, l.text)}>{l.text}</span>
            </text>
          ))
        )}
      </scrollbox>
      <box
        style={{
          height: 1,
          flexShrink: 0,
          flexDirection: "row",
          justifyContent: "space-between",
          paddingLeft: 1,
          paddingRight: 1,
        }}
      >
        <text fg={c.faint}>{`${lines.length} lines`}</text>
        <text fg={follow ? c.ok : c.faint}>{follow ? "following" : "paused"}</text>
      </box>
    </box>
  )
}
