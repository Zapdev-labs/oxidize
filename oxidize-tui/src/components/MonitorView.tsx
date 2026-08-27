import { useStore } from "../state/store.js"
import { c } from "../theme.js"
import { bytes, ms, num } from "../util/format.js"
import { Meter, Sparkline, Stat } from "./primitives.js"

function Card({
  title,
  children,
  width,
}: {
  title: string
  children: React.ReactNode
  width?: number | `${number}%`
}) {
  return (
    <box
      style={{
        flexGrow: width ? 0 : 1,
        width,
        flexDirection: "column",
        backgroundColor: c.panel,
        border: true,
        borderColor: c.border,
        paddingLeft: 1,
        paddingRight: 1,
      }}
      title={` ${title} `}
      titleColor={c.faint}
    >
      {children}
    </box>
  )
}

export function MonitorView({ width }: { width: number }) {
  const { latest, history, error } = useStore((s) => s.metrics)
  const server = useStore((s) => s.server)

  // Cards are laid out on an explicit grid so meters and sparklines can be sized
  // to their own card rather than to the terminal.
  const grid = width - 2 // view padding
  const inner = (cardWidth: number) => Math.max(8, cardWidth - 4) // border + padding
  const half = Math.floor((grid - 1) / 2)
  const tokensW = 24
  const third = Math.floor((grid - tokensW - 2) / 2)

  if (!server.url) {
    return (
      <box style={{ flexGrow: 1, padding: 2, flexDirection: "column" }}>
        <text fg={c.dim}>No server attached.</text>
        <text fg={c.faint}>Load a model (press 2) or attach with --api http://host:port.</text>
      </box>
    )
  }

  const uptime = server.startedAt ? Date.now() - server.startedAt : 0
  const kvRatio =
    latest && latest.kvBlocksTotal > 0 ? latest.kvBlocksUsed / latest.kvBlocksTotal : 0

  return (
    <box style={{ flexGrow: 1, flexDirection: "column", padding: 1, gap: 0 }}>
      <box style={{ flexDirection: "row", gap: 1 }}>
        <Card title="throughput" width={half}>
          <box style={{ flexDirection: "row", justifyContent: "space-between" }}>
            <Stat label="tok/s" value={num(latest?.tokensPerSecond ?? 0, 1)} color={c.accent} />
            <Stat label="mean infer" value={ms((latest?.inferenceMean ?? 0) * 1000)} color={c.text} />
          </box>
          <Sparkline data={history.map((h) => h.tokensPerSecond)} width={inner(half)} />
        </Card>
        <Card title="load" width={grid - half - 1}>
          <box style={{ flexDirection: "row", justifyContent: "space-between" }}>
            <Stat label="in flight" value={String(latest?.requestsInFlight ?? 0)} color={c.steel} />
            <Stat label="queued" value={String(latest?.queueDepth ?? 0)} color={c.steel} />
            <Stat label="requests" value={String(latest?.requestsTotal ?? 0)} />
          </box>
          <Sparkline
            data={history.map((h) => h.requestsInFlight + h.queueDepth)}
            width={inner(grid - half - 1)}
            color={c.steel}
          />
        </Card>
      </box>

      <box style={{ flexDirection: "row", gap: 1 }}>
        <Card title="tokens" width={tokensW}>
          <Stat label="generated" value={(latest?.tokensGenerated ?? 0).toLocaleString()} color={c.accent} />
          <Stat label="prompt" value={(latest?.tokensPrompt ?? 0).toLocaleString()} />
        </Card>
        <Card title="kv cache" width={third}>
          <Stat
            label="blocks"
            value={`${latest?.kvBlocksUsed ?? 0} / ${latest?.kvBlocksTotal ?? 0}`}
            color={kvRatio > 0.9 ? c.warn : c.text}
          />
          <Meter value={kvRatio} width={inner(third)} color={kvRatio > 0.9 ? c.warn : c.accent} />
          <Stat label="bytes" value={bytes(latest?.kvCacheBytes ?? 0)} />
        </Card>
        <Card title="prefix cache" width={grid - tokensW - third - 2}>
          <Stat label="hit ratio" value={`${num((latest?.cacheHitRatio ?? 0) * 100, 0)}%`} color={c.ok} />
          <Meter value={latest?.cacheHitRatio ?? 0} width={inner(grid - tokensW - third - 2)} color={c.ok} />
        </Card>
      </box>

      <box style={{ flexDirection: "row", gap: 1 }}>
        <Card title="process">
          <box style={{ flexDirection: "row", justifyContent: "space-between" }}>
            <Stat label="uptime" value={ms(uptime)} />
            <Stat label="pid" value={server.pid ? String(server.pid) : server.external ? "ext" : "-"} />
            <Stat label="errors" value={String(latest?.errorsTotal ?? 0)} color={(latest?.errorsTotal ?? 0) > 0 ? c.err : c.text} />
            <Stat label="ws conns" value={String(latest?.realtimeConnections ?? 0)} />
            <Stat label="endpoint" value={server.url.replace("http://", "")} color={c.steel} />
          </box>
        </Card>
      </box>

      {error ? <text fg={c.err}>{`metrics: ${error}`}</text> : null}
    </box>
  )
}
