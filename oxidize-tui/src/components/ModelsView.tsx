import { useMemo } from "react"

import { formatValue, ggmlTypeName, type GgufHeader } from "../engine/gguf.js"
import { useStore, type State } from "../state/store.js"
import { c, glyph } from "../theme.js"
import { bytes, ellipsizeStart, fuzzy, pad, padStart, params as fmtParams } from "../util/format.js"

export function filteredModels(s: State) {
  const q = s.models.filter.trim()
  if (!q) return s.models.entries
  return s.models.entries
    .map((e) => ({ e, score: fuzzy(q, `${e.name} ${e.facts?.arch ?? ""} ${e.facts?.quant ?? ""}`) }))
    .filter((x) => x.score != null)
    .sort((a, b) => (a.score! - b.score!))
    .map((x) => x.e)
}

function Row({
  cols,
  selected,
  loaded,
  cells,
}: {
  cols: number[]
  selected: boolean
  loaded: boolean
  cells: string[]
}) {
  const fg = selected ? c.bg : loaded ? c.accent : c.text
  const bg = selected ? c.accent : undefined
  return (
    <text bg={bg} fg={fg} wrapMode="none" truncate>
      <span fg={selected ? c.bg : loaded ? c.accent : c.faint}>{loaded ? " ▸ " : "   "}</span>
      {cells.map((cell, i) => {
        const w = cols[i]!
        return (
          <span key={i} fg={selected ? c.bg : i === 0 ? fg : c.dim}>
            {(i === 0 ? "" : " ") + (i === 0 ? pad(cell, w) : padStart(cell, w))}
          </span>
        )
      })}
    </text>
  )
}

function Inspect({ path, header, error }: { path: string; header?: GgufHeader; error?: string }) {
  const meta = header ? [...header.metadata.entries()] : []

  return (
    <box
      style={{
        flexGrow: 1,
        flexDirection: "column",
        backgroundColor: c.panel,
        border: true,
        borderColor: c.borderHot,
        paddingLeft: 1,
        paddingRight: 1,
      }}
      title={` ${ellipsizeStart(path, 70)} `}
      titleColor={c.accent}
      bottomTitle=" esc close · ↑↓ scroll "
      bottomTitleAlignment="right"
    >
      {error ? (
        <text fg={c.err}>{error}</text>
      ) : !header ? (
        <text fg={c.dim}>reading header…</text>
      ) : (
        <scrollbox
          focused
          style={{
            flexGrow: 1,
            rootOptions: { backgroundColor: c.panel },
            viewportOptions: { backgroundColor: c.panel },
            contentOptions: { backgroundColor: c.panel },
          }}
        >
          <text fg={c.faint}>{`GGUF v${header.version} · ${header.tensorCount} tensors · ${meta.length} metadata keys`}</text>
          <box style={{ height: 1 }} />
          <text fg={c.steel}>METADATA</text>
          {meta.map(([k, v]) => (
            <text key={k} wrapMode="none" truncate>
              <span fg={c.dim}>{pad(k, 42)}</span>
              <span fg={c.text}>{formatValue(v)}</span>
            </text>
          ))}
          <box style={{ height: 1 }} />
          <text fg={c.steel}>TENSORS</text>
          {header.tensors.map((t) => (
            <text key={t.name} wrapMode="none" truncate>
              <span fg={c.dim}>{pad(t.name, 42)}</span>
              <span fg={c.accent}>{pad(ggmlTypeName(t.ggmlType), 9)}</span>
              <span fg={c.text}>{`[${t.dims.join(" × ")}]`}</span>
            </text>
          ))}
        </scrollbox>
      )}
    </box>
  )
}

export function ModelsView({ height, width }: { height: number; width: number }) {
  const models = useStore((s) => s.models)
  const loadedPath = useStore((s) => s.server.modelPath)
  const rows = useStore(filteredModels)

  const { cols, keys, headers } = useMemo(() => {
    const all = [
      { key: "size", header: "SIZE", w: 7 },
      { key: "quant", header: "QUANT", w: 8 },
      { key: "params", header: "PARAMS", w: 8 },
      { key: "arch", header: "ARCH", w: 12 },
      { key: "ctx", header: "CTX", w: 9 },
      { key: "source", header: "SOURCE", w: 10 },
    ] as const
    const content = Math.max(1, width - 2 - 3)
    const minName = Math.min(28, Math.max(8, content))
    let kept = all.length
    while (kept > 0 && content - all.slice(0, kept).reduce((a, x) => a + x.w + 1, 0) < minName) kept--
    const shown = all.slice(0, kept)
    const tail = shown.reduce((a, x) => a + x.w + 1, 0)
    return {
      cols: [Math.max(1, content - tail), ...shown.map((x) => x.w)],
      keys: shown.map((x) => x.key as string),
      headers: ["NAME", ...shown.map((x) => x.header)],
    }
  }, [width])

  if (models.inspect) {
    return <Inspect {...models.inspect} />
  }

  const bodyHeight = Math.max(1, height - 4)
  const cursor = Math.min(models.cursor, Math.max(0, rows.length - 1))
  const start = Math.max(0, Math.min(cursor - Math.floor(bodyHeight / 2), rows.length - bodyHeight))
  const window = rows.slice(Math.max(0, start), Math.max(0, start) + bodyHeight)

  return (
    <box
      style={{
        flexGrow: 1,
        flexDirection: "column",
        backgroundColor: c.bg,
        paddingLeft: 1,
        paddingRight: 1,
      }}
    >
      <box style={{ flexDirection: "row", justifyContent: "space-between", height: 1 }}>
        <text wrapMode="none" truncate>
          <span fg={c.faint}>{"   "}</span>
          {headers.map((h, i) => (
            <span key={h} fg={c.faint}>
              {(i === 0 ? "" : " ") + (i === 0 ? pad(h, cols[i]!) : padStart(h, cols[i]!))}
            </span>
          ))}
        </text>
      </box>

      <box style={{ flexGrow: 1, flexDirection: "column" }}>
        {models.scanning && rows.length === 0 ? (
          <text fg={c.dim}>scanning…</text>
        ) : rows.length === 0 ? (
          <box style={{ flexDirection: "column", paddingTop: 1 }}>
            <text fg={c.dim}>No GGUF files found.</text>
            <text fg={c.faint}>
              Searched ./models, ~/.cache/oxidize/hf and $OXIDIZE_MODELS. Press p to pull one from
              Hugging Face, or r to rescan.
            </text>
          </box>
        ) : (
          window.map((e, i) => {
            const idx = Math.max(0, start) + i
            const cell: Record<string, string> = {
              size: bytes(e.size),
              quant: e.facts?.quant ?? (e.factsError ? "err" : "…"),
              params: e.facts ? fmtParams(e.facts.params) : "",
              arch: e.facts?.arch ?? "",
              ctx: e.facts?.ctx ? e.facts.ctx.toLocaleString() : "",
              source: e.source,
            }
            return (
              <Row
                key={e.path}
                cols={cols}
                selected={idx === cursor}
                loaded={e.path === loadedPath}
                cells={[
                  e.shards > 1 ? `${e.name}  (${e.shards} shards)` : e.name,
                  ...keys.map((k) => cell[k] ?? ""),
                ]}
              />
            )
          })
        )}
      </box>

      <box style={{ height: 1, flexDirection: "row", justifyContent: "space-between" }}>
        {models.filtering ? (
          <text>
            <span fg={c.accent}>{"/"}</span>
            <span fg={c.text}>{models.filter}</span>
            <span fg={c.accent}>{glyph.bar}</span>
          </text>
        ) : (
          <text fg={c.faint}>
            {`${rows.length}/${models.entries.length} models${models.scanning ? " · scanning" : ""}`}
          </text>
        )}
        <text fg={c.faint}>{loadedPath ? `loaded ${ellipsizeStart(loadedPath, 48)}` : ""}</text>
      </box>
    </box>
  )
}
