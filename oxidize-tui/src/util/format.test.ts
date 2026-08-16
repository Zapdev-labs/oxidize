import { describe, expect, test } from "bun:test"

import { ellipsizeStart, pad, padStart } from "./format.js"

describe("terminal-width formatting", () => {
  test("truncates wide graphemes without exceeding the cell budget", () => {
    expect(pad("模型", 3)).toBe("模…")
    expect(padStart("模型", 3)).toBe("模…")
    expect(ellipsizeStart("/models/模型.gguf", 8)).toBe("…型.gguf")
  })
})
