/**
 * Instrument-panel palette. One accent (oxide orange), one secondary (steel),
 * everything else is neutral. No gradients, no decoration.
 */
export const c = {
  bg: "#0b0c0e",
  panel: "#101216",
  panelAlt: "#15181d",
  raised: "#1b1f25",

  border: "#23272e",
  borderHot: "#3b424c",

  text: "#d3d7dd",
  dim: "#7c8390",
  faint: "#4b515b",

  accent: "#ff6a1f",
  accentDim: "#a3461c",
  steel: "#5aa9bd",
  steelDim: "#2f6070",

  ok: "#69b578",
  warn: "#d6a533",
  err: "#d9584a",
} as const

export const glyph = {
  bar: "▌",
  dot: "●",
  arrow: "›",
  spark: ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"] as const,
  spinner: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const,
} as const

export const roleColor = (role: string) =>
  role === "user" ? c.steel : role === "assistant" ? c.accent : c.dim
