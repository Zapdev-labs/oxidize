import type { InputRenderable } from "@opentui/core"
import { useEffect, useRef } from "react"

import { set } from "../state/store.js"
import { c } from "../theme.js"

export function Prompt({
  label,
  placeholder,
  submit,
  width,
}: {
  label: string
  placeholder: string
  submit: (value: string) => void
  width: number
}) {
  const inputRef = useRef<InputRenderable>(null)
  useEffect(() => {
    inputRef.current?.focus?.()
  }, [])

  const boxWidth = Math.min(70, Math.max(40, width - 8))
  return (
    <box
      style={{
        position: "absolute",
        left: Math.max(2, Math.floor((width - boxWidth) / 2)),
        top: 4,
        width: boxWidth,
        height: 3,
        zIndex: 100,
        backgroundColor: c.panelAlt,
        border: true,
        borderColor: c.accentDim,
        paddingLeft: 1,
        paddingRight: 1,
      }}
      title={` ${label} `}
      titleColor={c.accent}
      bottomTitle=" ⏎ confirm · esc cancel "
      bottomTitleAlignment="right"
    >
      <input
        ref={inputRef}
        focused
        placeholder={placeholder}
        textColor={c.text}
        placeholderColor={c.faint}
        cursorColor={c.accent}
        backgroundColor={c.panelAlt}
        onSubmit={(submitted) => {
          const value = typeof submitted === "string" ? submitted : (inputRef.current?.value ?? "")
          set({ overlay: null })
          submit(value)
        }}
      />
    </box>
  )
}
