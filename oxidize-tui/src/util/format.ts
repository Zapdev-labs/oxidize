export function bytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "-"
  const units = ["B", "K", "M", "G", "T"]
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)}${units[i]}`
}

export function params(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "-"
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`
  return `${(n / 1e3).toFixed(0)}K`
}

export function ms(n: number | undefined): string {
  if (n == null || !Number.isFinite(n)) return "-"
  if (n < 1000) return `${n.toFixed(0)}ms`
  if (n < 60_000) return `${(n / 1000).toFixed(1)}s`
  const m = Math.floor(n / 60_000)
  return `${m}m${Math.floor((n % 60_000) / 1000)}s`
}

export function num(n: number | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return "-"
  return n.toFixed(digits)
}

export function clock(t: number): string {
  const d = new Date(t)
  const p = (v: number) => v.toString().padStart(2, "0")
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

export function pad(s: string, width: number): string {
  const clipped = ellipsizeEnd(s, width)
  return clipped + " ".repeat(Math.max(0, width - displayWidth(clipped)))
}

export function padStart(s: string, width: number): string {
  const clipped = ellipsizeEnd(s, width)
  return " ".repeat(Math.max(0, width - displayWidth(clipped))) + clipped
}

export function ellipsizeStart(s: string, width: number): string {
  if (displayWidth(s) <= width) return s
  if (width <= 0) return ""
  if (width === 1) return "…"
  const tail: string[] = []
  let used = 1
  for (const grapheme of [...new Intl.Segmenter().segment(s)].map((part) => part.segment).reverse()) {
    const cells = graphemeWidth(grapheme)
    if (used + cells > width) break
    tail.unshift(grapheme)
    used += cells
  }
  return "…" + tail.join("")
}

function ellipsizeEnd(s: string, width: number): string {
  if (displayWidth(s) <= width) return s
  if (width <= 0) return ""
  if (width === 1) return "…"
  const head: string[] = []
  let used = 1
  for (const part of new Intl.Segmenter().segment(s)) {
    const cells = graphemeWidth(part.segment)
    if (used + cells > width) break
    head.push(part.segment)
    used += cells
  }
  return head.join("") + "…"
}

function displayWidth(s: string): number {
  return [...new Intl.Segmenter().segment(s)].reduce((width, part) => width + graphemeWidth(part.segment), 0)
}

function graphemeWidth(grapheme: string): number {
  if (/^[\p{Mark}\p{Control}\u200d\ufe0f]*$/u.test(grapheme)) return 0
  return /[\p{Extended_Pictographic}\p{Script=Han}\p{Script=Hangul}\p{Script=Hiragana}\p{Script=Katakana}]/u.test(grapheme)
    ? 2
    : 1
}

/** Subsequence fuzzy match; returns a score (lower = better) or null. */
export function fuzzy(needle: string, haystack: string): number | null {
  if (!needle) return 0
  const n = needle.toLowerCase()
  const h = haystack.toLowerCase()
  let hi = 0
  let score = 0
  let lastHit = -1
  for (const ch of n) {
    const found = h.indexOf(ch, hi)
    if (found < 0) return null
    score += found - (lastHit + 1)
    lastHit = found
    hi = found + 1
  }
  return score + haystack.length * 0.01
}
