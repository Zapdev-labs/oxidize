import { describe, expect, test } from "bun:test"
import { mkdtemp, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"

import { factsFrom, ggmlTypeName, readGgufHeader } from "./gguf.js"

/** Minimal GGUF v3 writer, only rich enough to exercise the reader. */
class Writer {
  private parts: Buffer[] = []
  push(b: Buffer) {
    this.parts.push(b)
    return this
  }
  u32(v: number) {
    const b = Buffer.alloc(4)
    b.writeUInt32LE(v)
    return this.push(b)
  }
  u64(v: number) {
    const b = Buffer.alloc(8)
    b.writeBigUInt64LE(BigInt(v))
    return this.push(b)
  }
  f32(v: number) {
    const b = Buffer.alloc(4)
    b.writeFloatLE(v)
    return this.push(b)
  }
  str(s: string) {
    const body = Buffer.from(s, "utf8")
    return this.u64(body.length).push(body)
  }
  done() {
    return Buffer.concat(this.parts)
  }
}

function buildModel(): Buffer {
  const w = new Writer()
  w.push(Buffer.from("GGUF", "ascii")).u32(3)
  w.u64(2) // tensors
  w.u64(6) // kv pairs

  w.str("general.architecture").u32(8).str("llama")
  w.str("general.name").u32(8).str("Test Model")
  w.str("llama.context_length").u32(4).u32(4096)
  w.str("llama.block_count").u32(4).u32(2)
  w.str("llama.attention.head_count").u32(4).u32(8)
  // A big array must be walked but not retained.
  w.str("tokenizer.ggml.tokens").u32(9).u32(8).u64(600)
  for (let i = 0; i < 600; i++) w.str(`t${i}`)

  // blk.* dominated by Q4_K (12) so the quant label must resolve to Q4_K
  w.str("blk.0.attn_q.weight").u32(2).u64(4096).u64(4096).u32(12).u64(0)
  w.str("output.weight").u32(2).u64(4096).u64(32000).u32(0).u64(1024)
  return w.done()
}

describe("gguf reader", () => {
  test("parses the oxidize-core v3 fixture", async () => {
    const header = await readGgufHeader("../oxidize-core/tests/fixtures/valid-v3.gguf")
    expect(header.version).toBe(3)
    expect(header.tensorCount).toBe(1)
    expect(header.metadata.get("general.alignment")).toBe(64)
    expect(header.tensors[0]!.name).toBe("tok_embeddings.weight")
    expect(header.tensors[0]!.dims).toEqual([32000, 4096])
  })

  test("rejects a bad magic", async () => {
    await expect(
      readGgufHeader("../oxidize-core/tests/fixtures/invalid-magic.gguf"),
    ).rejects.toThrow(/not a GGUF file/)
  })

  test("rejects an unsupported version", async () => {
    await expect(
      readGgufHeader("../oxidize-core/tests/fixtures/unsupported-version.gguf"),
    ).rejects.toThrow(/unsupported GGUF version/)
  })

  test("derives facts, including quant from the dominant block tensor", async () => {
    const dir = await mkdtemp(join(tmpdir(), "oxidize-tui-"))
    const path = join(dir, "model.gguf")
    await writeFile(path, buildModel())

    const header = await readGgufHeader(path)
    const facts = factsFrom(header, "fallback")

    expect(facts.arch).toBe("llama")
    expect(facts.name).toBe("Test Model")
    expect(facts.ctx).toBe(4096)
    expect(facts.layers).toBe(2)
    expect(facts.heads).toBe(8)
    expect(facts.quant).toBe("Q4_K")
    // 4096*4096 + 4096*32000
    expect(facts.params).toBe(4096 * 4096 + 4096 * 32000)
    // Large arrays are summarised, not retained.
    expect(String(header.metadata.get("tokenizer.ggml.tokens"))).toContain("600")
  })

  test("names oxidize's AL family", () => {
    expect(ggmlTypeName(240)).toBe("AL5")
    expect(ggmlTypeName(243)).toBe("AL5_XS")
    expect(ggmlTypeName(999)).toBe("T999")
  })
})
