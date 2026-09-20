import { describe, expect, test } from "bun:test"
import { resolve } from "node:path"

import { findWorkspaceRoot } from "./binary.js"

describe("workspace root", () => {
  test("finds the C + TUI workspace from oxidize-tui", () => {
    const root = findWorkspaceRoot(import.meta.dir)
    expect(root).not.toBeNull()
    expect(resolve(root!)).toBe(resolve(import.meta.dir, "../../.."))
  })
})
