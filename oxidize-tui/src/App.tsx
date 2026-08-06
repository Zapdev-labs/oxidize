import type { KeyEvent } from "@opentui/core"
import { useKeyboard, useTerminalDimensions } from "@opentui/react"
import { useEffect, useState } from "react"

import { ChatView } from "./components/ChatView.js"
import { Header, StatusBar, Tabs } from "./components/Chrome.js"
import { LogsView } from "./components/LogsView.js"
import { filteredModels, ModelsView } from "./components/ModelsView.js"
import { MonitorView } from "./components/MonitorView.js"
import { Help, Palette, paletteMatches } from "./components/Palette.js"
import { Prompt } from "./components/Prompt.js"
import {
  cancelGeneration,
  clearChat,
  loadModel,
  pullModel,
  rescanModels,
  shutdown,
} from "./state/actions.js"
import { getState, set, setIn, toast, useStore, type View } from "./state/store.js"
import { c, glyph } from "./theme.js"

const VIEW_ORDER: View[] = ["chat", "models", "monitor", "logs"]

const HINTS: Record<View, [string, string][]> = {
  chat: [
    ["⏎", "send"],
    ["esc", "cancel"],
    ["ctrl+t", "views"],
  ],
  models: [
    ["⏎", "load"],
    ["i", "inspect"],
    ["/", "filter"],
    ["p", "pull"],
    ["r", "rescan"],
  ],
  monitor: [["ctrl+t", "views"]],
  logs: [
    ["f", "follow"],
    ["c", "clear"],
  ],
}

function useSpinner(active: boolean) {
  const [frame, setFrame] = useState(0)
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setFrame((f) => f + 1), 90)
    return () => clearInterval(t)
  }, [active])
  return glyph.spinner[frame % glyph.spinner.length]!
}

export function App() {
  const { width, height } = useTerminalDimensions()
  const view = useStore((s) => s.view)
  const overlay = useStore((s) => s.overlay)
  const busy = useStore(
    (s) => s.chat.generating || s.server.status === "loading" || s.server.status === "starting",
  )
  const spin = useSpinner(busy)

  // Nudge a re-render each second so clocks and uptime stay honest.
  const [, tick] = useState(0)
  useEffect(() => {
    const t = setInterval(() => tick((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [])

  useKeyboard((key: KeyEvent) => {
    const s = getState()

    if (s.overlay?.kind === "palette") {
      const matches = paletteMatches(s.overlay.query)
      if (key.name === "escape") return set({ overlay: null })
      if (key.name === "up" || (key.ctrl && key.name === "p")) {
        return set({
          overlay: { ...s.overlay, cursor: Math.max(0, s.overlay.cursor - 1) },
        })
      }
      if (key.name === "down" || (key.ctrl && key.name === "n")) {
        return set({
          overlay: { ...s.overlay, cursor: Math.min(matches.length - 1, s.overlay.cursor + 1) },
        })
      }
      if (key.name === "return") {
        const cmd = matches[Math.min(s.overlay.cursor, matches.length - 1)]
        set({ overlay: null })
        cmd?.run()
      }
      return
    }

    if (s.overlay?.kind === "help") {
      if (key.name === "escape" || key.name === "return" || key.name === "q") set({ overlay: null })
      return
    }

    if (s.overlay?.kind === "prompt") {
      if (key.name === "escape") set({ overlay: null })
      return
    }

    if (key.ctrl && key.name === "c") {
      shutdown()
      process.exit(0)
    }
    if (key.ctrl && key.name === "k") {
      return set({ overlay: { kind: "palette", query: "", cursor: 0 } })
    }
    if (key.ctrl && key.name === "t") {
      const idx = VIEW_ORDER.indexOf(s.view)
      return set({ view: VIEW_ORDER[(idx + 1) % VIEW_ORDER.length]! })
    }
    if (key.ctrl && key.name === "l" && s.view === "chat") return clearChat()

    if (key.name === "escape") {
      if (s.chat.generating) {
        cancelGeneration()
        return toast("cancelled")
      }
      if (s.models.inspect) return setIn("models", { inspect: null })
      if (s.models.filtering) return setIn("models", { filtering: false, filter: "" })
      return
    }

    // Below here: single-key bindings, only where no text field owns the keyboard.
    if (s.view === "chat") return
    if (s.view === "models" && s.models.filtering) {
      if (key.name === "return") return setIn("models", { filtering: false })
      if (key.name === "backspace") {
        return setIn("models", (m) => ({ filter: m.filter.slice(0, -1) }))
      }
      if (key.sequence && key.sequence.length === 1 && !key.ctrl && !key.meta) {
        return setIn("models", (m) => ({ filter: m.filter + key.sequence, cursor: 0 }))
      }
      return
    }

    const jump = "1234".indexOf(key.name)
    if (jump >= 0 && !key.ctrl) return set({ view: VIEW_ORDER[jump]! })
    if (key.name === "?") return set({ overlay: { kind: "help" } })
    if (key.name === "q") {
      shutdown()
      process.exit(0)
    }

    if (s.view === "models") {
      const rows = filteredModels(s)
      const move = (delta: number) =>
        setIn("models", (m) => ({
          cursor: Math.max(0, Math.min(rows.length - 1, m.cursor + delta)),
        }))
      if (s.models.inspect) return
      switch (key.name) {
        case "j":
        case "down":
          return move(1)
        case "k":
        case "up":
          return move(-1)
        case "pagedown":
          return move(10)
        case "pageup":
          return move(-10)
        case "g":
          return setIn("models", { cursor: 0 })
        case "G":
          return setIn("models", { cursor: Math.max(0, rows.length - 1) })
        case "return": {
          const entry = rows[Math.min(s.models.cursor, rows.length - 1)]
          if (entry) void loadModel(entry.path)
          return
        }
        case "i": {
          const entry = rows[Math.min(s.models.cursor, rows.length - 1)]
          if (entry) setIn("models", { inspect: entry.path })
          return
        }
        case "/":
          return setIn("models", { filtering: true, filter: "" })
        case "r":
          return void rescanModels()
        case "p":
          return set({
            overlay: {
              kind: "prompt",
              label: "pull from Hugging Face",
              placeholder: "owner/repo-GGUF",
              submit: (v) => void pullModel(v),
            },
          })
      }
      return
    }

    if (s.view === "logs") {
      if (key.name === "f") return setIn("logs", (l) => ({ follow: !l.follow }))
      if (key.name === "c") return setIn("logs", { lines: [] })
    }
  })

  const bodyHeight = Math.max(1, height - 3)

  return (
    <box style={{ width, height, flexDirection: "column", backgroundColor: c.bg }}>
      <Header width={width} spin={spin} />
      <Tabs active={view} />

      <box style={{ flexGrow: 1, flexDirection: "column" }}>
        {view === "chat" ? (
          <ChatView focused={overlay === null} width={width} spin={spin} />
        ) : view === "models" ? (
          <ModelsView height={bodyHeight} width={width} />
        ) : view === "monitor" ? (
          <MonitorView width={width} />
        ) : (
          <LogsView focused={overlay === null} />
        )}
      </box>

      <StatusBar hints={HINTS[view]} width={width} />

      {overlay?.kind === "palette" ? (
        <Palette query={overlay.query} cursor={overlay.cursor} width={width} />
      ) : overlay?.kind === "help" ? (
        <Help width={width} />
      ) : overlay?.kind === "prompt" ? (
        <Prompt
          label={overlay.label}
          placeholder={overlay.placeholder}
          submit={overlay.submit}
          width={width}
        />
      ) : null}
    </box>
  )
}
