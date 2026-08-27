import type { ReactNode } from "react"

import { c, glyph } from "../theme.js"

export function Sparkline({
  data,
  width,
  color = c.accent,
  max,
}: {
  data: number[]
  width: number
  color?: string
  max?: number
}) {
  if (width <= 0) return null
  const window = data.slice(-width)
  const peak = max ?? Math.max(...window, 1e-9)
  const cells = window.map((v) => {
    if (!Number.isFinite(v) || v <= 0) return glyph.spark[0]
    const idx = Math.min(glyph.spark.length - 1, Math.round((v / peak) * (glyph.spark.length - 1)))
    return glyph.spark[idx]
  })
  const pad = " ".repeat(Math.max(0, width - cells.length))
  return (
    <text fg={color}>
      {pad}
      {cells.join("")}
    </text>
  )
}

export function Meter({ value, width, color = c.accent }: { value: number; width: number; color?: string }) {
  const filled = Math.max(0, Math.min(width, Math.round(value * width)))
  return (
    <text>
      <span fg={color}>{"█".repeat(filled)}</span>
      <span fg={c.faint}>{"░".repeat(Math.max(0, width - filled))}</span>
    </text>
  )
}

export function Stat({
  label,
  value,
  unit,
  color = c.text,
}: {
  label: string
  value: string
  unit?: string
  color?: string
}) {
  return (
    <box style={{ flexDirection: "column" }}>
      <text fg={c.faint}>{label.toUpperCase()}</text>
      <text>
        <span fg={color}>{value}</span>
        {unit ? <span fg={c.faint}>{` ${unit}`}</span> : null}
      </text>
    </box>
  )
}

export function Field({ k, v, color = c.text }: { k: string; v: string; color?: string }) {
  return (
    <text>
      <span fg={c.faint}>{k.padEnd(11)}</span>
      <span fg={color}>{v}</span>
    </text>
  )
}

export function Panel({
  title,
  right,
  children,
  focused,
  grow = true,
}: {
  title: string
  right?: string
  children: ReactNode
  focused?: boolean
  grow?: boolean
}) {
  return (
    <box
      style={{
        border: true,
        borderStyle: "single",
        borderColor: focused ? c.borderHot : c.border,
        backgroundColor: c.panel,
        flexDirection: "column",
        flexGrow: grow ? 1 : 0,
        paddingLeft: 1,
        paddingRight: 1,
      }}
      title={` ${title} `}
      titleColor={focused ? c.accent : c.dim}
      bottomTitle={right ? ` ${right} ` : undefined}
      bottomTitleAlignment="right"
    >
      {children}
    </box>
  )
}

export function Keys({ items }: { items: [string, string][] }) {
  return (
    <text>
      {items.map(([key, desc], i) => (
        <span key={key + i}>
          <span fg={c.dim}>{key}</span>
          <span fg={c.faint}>{` ${desc}`}</span>
          {i < items.length - 1 ? <span fg={c.faint}>{"   "}</span> : null}
        </span>
      ))}
    </text>
  )
}
