/**
 * Markdown-lite renderer. Deliberately not a full parser: model output uses a
 * small, predictable subset and a real parser costs more than it returns here.
 */
import { Fragment, type ReactNode } from "react"

import { c } from "../theme.js"

type Block =
  | { kind: "p"; text: string }
  | { kind: "h"; level: number; text: string }
  | { kind: "li"; marker: string; text: string; indent: number }
  | { kind: "quote"; text: string }
  | { kind: "code"; lang: string; lines: string[] }
  | { kind: "rule" }

const LIST_RE = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/
const HEAD_RE = /^(#{1,6})\s+(.*)$/

export function parseBlocks(src: string): Block[] {
  const lines = src.split("\n")
  const out: Block[] = []
  let para: string[] = []

  const flush = () => {
    if (para.length) {
      out.push({ kind: "p", text: para.join(" ") })
      para = []
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!
    const fence = /^\s*```(.*)$/.exec(line)
    if (fence) {
      flush()
      const lang = fence[1]!.trim()
      const body: string[] = []
      i++
      while (i < lines.length && !/^\s*```/.test(lines[i]!)) body.push(lines[i]!), i++
      out.push({ kind: "code", lang, lines: body })
      continue
    }
    if (!line.trim()) {
      flush()
      continue
    }
    if (/^\s*([-*_])\s*\1\s*\1[\s-*_]*$/.test(line)) {
      flush()
      out.push({ kind: "rule" })
      continue
    }
    const head = HEAD_RE.exec(line)
    if (head) {
      flush()
      out.push({ kind: "h", level: head[1]!.length, text: head[2]! })
      continue
    }
    const li = LIST_RE.exec(line)
    if (li) {
      flush()
      out.push({
        kind: "li",
        indent: Math.floor(li[1]!.length / 2),
        marker: /\d/.test(li[2]!) ? li[2]! : "•",
        text: li[3]!,
      })
      continue
    }
    if (/^\s*>\s?/.test(line)) {
      flush()
      out.push({ kind: "quote", text: line.replace(/^\s*>\s?/, "") })
      continue
    }
    para.push(line.trim())
  }
  flush()
  return out
}

/** `**bold**`, `*em*`, `` `code` `` — first match wins, no nesting. */
export function inline(src: string, key = "i"): ReactNode[] {
  const nodes: ReactNode[] = []
  const re = /(\*\*[^*]+\*\*|`[^`]+`|(?<![\w*])\*[^*\n]+\*(?![\w*])|_[^_\n]+_)/g
  let last = 0
  let m: RegExpExecArray | null
  let n = 0
  while ((m = re.exec(src))) {
    if (m.index > last) nodes.push(src.slice(last, m.index))
    const tok = m[0]!
    const id = `${key}-${n++}`
    if (tok.startsWith("**")) {
      nodes.push(
        <strong key={id} fg={c.text}>
          {tok.slice(2, -2)}
        </strong>,
      )
    } else if (tok.startsWith("`")) {
      nodes.push(
        <span key={id} fg={c.accent} bg={c.raised}>
          {` ${tok.slice(1, -1)} `}
        </span>,
      )
    } else {
      nodes.push(
        <em key={id} fg={c.text}>
          {tok.slice(1, -1)}
        </em>,
      )
    }
    last = m.index + tok.length
  }
  if (last < src.length) nodes.push(src.slice(last))
  return nodes.length ? nodes : [src]
}

export function Markdown({ content, dim = false }: { content: string; dim?: boolean }) {
  const blocks = parseBlocks(content)
  const base = dim ? c.dim : c.text

  return (
    <box style={{ flexDirection: "column", width: "100%" }}>
      {blocks.map((b, i) => {
        switch (b.kind) {
          case "h":
            return (
              <text key={i} fg={b.level === 1 ? c.accent : c.steel} wrapMode="word">
                <strong>{b.text}</strong>
              </text>
            )
          case "li":
            return (
              <text key={i} fg={base} wrapMode="word">
                <span fg={c.faint}>{`${"  ".repeat(b.indent)}${b.marker} `}</span>
                {inline(b.text, `l${i}`)}
              </text>
            )
          case "quote":
            return (
              <text key={i} fg={c.dim} wrapMode="word">
                <span fg={c.faint}>{"▏ "}</span>
                {inline(b.text, `q${i}`)}
              </text>
            )
          case "rule":
            return (
              <text key={i} fg={c.faint}>
                {"─".repeat(24)}
              </text>
            )
          case "code":
            return (
              <box
                key={i}
                style={{
                  flexDirection: "column",
                  backgroundColor: c.panelAlt,
                  border: ["left"],
                  borderColor: c.accentDim,
                  paddingLeft: 1,
                  marginTop: 0,
                  width: "100%",
                }}
              >
                {b.lang ? <text fg={c.faint}>{b.lang}</text> : null}
                {b.lines.map((l, j) => (
                  <text key={j} fg={c.steel} wrapMode="none" truncate>
                    {l || " "}
                  </text>
                ))}
              </box>
            )
          default:
            return (
              <text key={i} fg={base} wrapMode="word">
                {inline(b.text, `p${i}`)}
              </text>
            )
        }
      })}
    </box>
  )
}

export const MarkdownFragment = Fragment
