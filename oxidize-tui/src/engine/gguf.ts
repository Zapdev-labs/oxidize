/**
 * Minimal GGUF header reader.
 *
 * Reads metadata KVs and tensor descriptors without mapping weights, so the
 * model browser can show architecture / quant / parameter count for a 200 GB
 * file in a few milliseconds. Mirrors `oxidize-core/src/format/gguf.rs`.
 */
import { open, type FileHandle } from "node:fs/promises"

const MAGIC = 0x46554747 // "GGUF" little-endian

export type GgufValue = number | bigint | boolean | string | GgufValue[]

export interface GgufTensor {
  name: string
  dims: number[]
  ggmlType: number
  offset: bigint
  elements: number
}

export interface GgufHeader {
  version: number
  tensorCount: number
  metadata: Map<string, GgufValue>
  tensors: GgufTensor[]
}

export interface ModelFacts {
  arch: string
  quant: string
  params: number
  ctx: number
  layers: number
  embed: number
  heads: number
  expertCount: number
  name: string
  tensorCount: number
  version: number
}

/** ggml_type -> label. Includes oxidize's AL family (240-243). */
const GGML_TYPE_NAMES: Record<number, string> = {
  0: "F32",
  1: "F16",
  2: "Q4_0",
  3: "Q4_1",
  6: "Q5_0",
  7: "Q5_1",
  8: "Q8_0",
  9: "Q8_1",
  10: "Q2_K",
  11: "Q3_K",
  12: "Q4_K",
  13: "Q5_K",
  14: "Q6_K",
  15: "Q8_K",
  16: "IQ2_XXS",
  17: "IQ2_XS",
  18: "IQ3_XXS",
  19: "IQ1_S",
  20: "IQ4_NL",
  21: "IQ3_S",
  22: "IQ2_S",
  23: "IQ4_XS",
  24: "I8",
  25: "I16",
  26: "I32",
  27: "I64",
  28: "F64",
  29: "IQ1_M",
  30: "BF16",
  40: "NVFP4",
  240: "AL5",
  241: "AL8",
  242: "AL6",
  243: "AL5_XS",
}

export const ggmlTypeName = (t: number) => GGML_TYPE_NAMES[t] ?? `T${t}`

/** Grows its window on demand so we never read more of the file than needed. */
class Cursor {
  private buf: Buffer
  private filled = 0
  private pos = 0

  constructor(
    private fh: FileHandle,
    private fileSize: number,
    initial = 1 << 20,
  ) {
    this.buf = Buffer.alloc(Math.min(initial, Math.max(fileSize, 1)))
  }

  get offset() {
    return this.pos
  }

  private async ensure(n: number) {
    const need = this.pos + n
    if (need <= this.filled) return
    if (need > this.fileSize) throw new Error("gguf: truncated header")
    if (need > this.buf.length) {
      let cap = this.buf.length || 1 << 16
      while (cap < need) cap *= 2
      const next = Buffer.alloc(Math.min(cap, this.fileSize))
      this.buf.copy(next, 0, 0, this.filled)
      this.buf = next
    }
    const want = Math.min(this.buf.length, this.fileSize) - this.filled
    const { bytesRead } = await this.fh.read(this.buf, this.filled, want, this.filled)
    if (bytesRead <= 0) throw new Error("gguf: unexpected eof")
    this.filled += bytesRead
    if (this.pos + n > this.filled) await this.ensure(n)
  }

  async u8() {
    await this.ensure(1)
    return this.buf.readUInt8(this.pos++)
  }
  async i8() {
    await this.ensure(1)
    return this.buf.readInt8(this.pos++)
  }
  async u16() {
    await this.ensure(2)
    const v = this.buf.readUInt16LE(this.pos)
    this.pos += 2
    return v
  }
  async i16() {
    await this.ensure(2)
    const v = this.buf.readInt16LE(this.pos)
    this.pos += 2
    return v
  }
  async u32() {
    await this.ensure(4)
    const v = this.buf.readUInt32LE(this.pos)
    this.pos += 4
    return v
  }
  async i32() {
    await this.ensure(4)
    const v = this.buf.readInt32LE(this.pos)
    this.pos += 4
    return v
  }
  async f32() {
    await this.ensure(4)
    const v = this.buf.readFloatLE(this.pos)
    this.pos += 4
    return v
  }
  async f64() {
    await this.ensure(8)
    const v = this.buf.readDoubleLE(this.pos)
    this.pos += 8
    return v
  }
  async u64() {
    await this.ensure(8)
    const v = this.buf.readBigUInt64LE(this.pos)
    this.pos += 8
    return v
  }
  async i64() {
    await this.ensure(8)
    const v = this.buf.readBigInt64LE(this.pos)
    this.pos += 8
    return v
  }
  async str() {
    const len = Number(await this.u64())
    if (len > 1 << 26) throw new Error("gguf: absurd string length")
    await this.ensure(len)
    const s = this.buf.toString("utf8", this.pos, this.pos + len)
    this.pos += len
    return s
  }
}

async function readValue(cur: Cursor, type: number, depth = 0): Promise<GgufValue> {
  switch (type) {
    case 0:
      return cur.u8()
    case 1:
      return cur.i8()
    case 2:
      return cur.u16()
    case 3:
      return cur.i16()
    case 4:
      return cur.u32()
    case 5:
      return cur.i32()
    case 6:
      return cur.f32()
    case 7:
      return (await cur.u8()) !== 0
    case 8:
      return cur.str()
    case 9: {
      if (depth > 4) throw new Error("gguf: array nesting too deep")
      const inner = await cur.u32()
      const count = Number(await cur.u64())
      const out: GgufValue[] = []
      // Tokenizer vocabularies run to hundreds of thousands of entries; we must
      // walk them to reach later keys but there is no reason to retain them.
      const keep = count <= 512
      for (let i = 0; i < count; i++) {
        const v = await readValue(cur, inner, depth + 1)
        if (keep) out.push(v)
      }
      return keep ? out : [`<${count} × ${typeName(inner)}>`]
    }
    case 10:
      return cur.u64()
    case 11:
      return cur.i64()
    case 12:
      return cur.f64()
    default:
      throw new Error(`gguf: unknown value type ${type}`)
  }
}

function typeName(t: number): string {
  return (
    ["u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "string", "array", "u64", "i64", "f64"][t] ??
    `type${t}`
  )
}

export async function readGgufHeader(path: string): Promise<GgufHeader> {
  const fh = await open(path, "r")
  try {
    const st = await fh.stat()
    const cur = new Cursor(fh, st.size)
    if ((await cur.u32()) !== MAGIC) throw new Error("not a GGUF file")
    const version = await cur.u32()
    if (version < 2 || version > 3) throw new Error(`unsupported GGUF version ${version}`)
    const tensorCount = Number(await cur.u64())
    const kvCount = Number(await cur.u64())
    if (kvCount > 1 << 20) throw new Error("gguf: absurd metadata count")

    const metadata = new Map<string, GgufValue>()
    for (let i = 0; i < kvCount; i++) {
      const key = await cur.str()
      const type = await cur.u32()
      metadata.set(key, await readValue(cur, type))
    }

    const tensors: GgufTensor[] = []
    for (let i = 0; i < tensorCount; i++) {
      const name = await cur.str()
      const nDims = await cur.u32()
      if (nDims > 8) throw new Error("gguf: absurd tensor rank")
      const dims: number[] = []
      for (let d = 0; d < nDims; d++) dims.push(Number(await cur.u64()))
      const ggmlType = await cur.u32()
      const offset = await cur.u64()
      tensors.push({
        name,
        dims,
        ggmlType,
        offset,
        elements: dims.reduce((a, b) => a * b, 1),
      })
    }

    return { version, tensorCount, metadata, tensors }
  } finally {
    await fh.close()
  }
}

const asNum = (v: GgufValue | undefined): number | undefined => {
  if (typeof v === "number") return v
  if (typeof v === "bigint") return Number(v)
  return undefined
}

const asStr = (v: GgufValue | undefined): string | undefined =>
  typeof v === "string" ? v : undefined

/** Pick the first key present, trying `<arch>.<suffix>` before generic keys. */
function archKey(md: Map<string, GgufValue>, arch: string, suffix: string): number | undefined {
  return asNum(md.get(`${arch}.${suffix}`)) ?? asNum(md.get(`general.${suffix}`))
}

/**
 * Quantization label. `general.file_type` is unreliable for custom types, so we
 * take the modal ggml type across the transformer block weights instead.
 */
function dominantQuant(tensors: GgufTensor[]): string {
  const weight = new Map<number, number>()
  for (const t of tensors) {
    if (!t.name.startsWith("blk.")) continue
    weight.set(t.ggmlType, (weight.get(t.ggmlType) ?? 0) + t.elements)
  }
  if (weight.size === 0) {
    for (const t of tensors) weight.set(t.ggmlType, (weight.get(t.ggmlType) ?? 0) + t.elements)
  }
  let best = -1
  let bestN = -1
  for (const [type, n] of weight) {
    if (n > bestN) {
      best = type
      bestN = n
    }
  }
  return best < 0 ? "-" : ggmlTypeName(best)
}

export function factsFrom(header: GgufHeader, fallbackName: string): ModelFacts {
  const md = header.metadata
  const arch = asStr(md.get("general.architecture")) ?? "unknown"
  const declared = asStr(md.get("general.parameter_count"))
  const params =
    asNum(md.get("general.parameter_count")) ??
    (declared ? Number(declared) : undefined) ??
    header.tensors.reduce((a, t) => a + t.elements, 0)

  return {
    arch,
    quant: dominantQuant(header.tensors),
    params,
    ctx: archKey(md, arch, "context_length") ?? 0,
    layers: archKey(md, arch, "block_count") ?? 0,
    embed: archKey(md, arch, "embedding_length") ?? 0,
    heads: archKey(md, arch, "attention.head_count") ?? 0,
    expertCount: archKey(md, arch, "expert_count") ?? 0,
    name: asStr(md.get("general.name")) ?? fallbackName,
    tensorCount: header.tensorCount,
    version: header.version,
  }
}

export function formatValue(v: GgufValue): string {
  if (Array.isArray(v)) {
    const head = v.slice(0, 8).map(formatValue).join(", ")
    return v.length > 8 ? `[${head}, …]` : `[${head}]`
  }
  if (typeof v === "bigint") return v.toString()
  if (typeof v === "number") return Number.isInteger(v) ? v.toString() : v.toFixed(6)
  if (typeof v === "boolean") return v ? "true" : "false"
  return v.length > 160 ? `${v.slice(0, 160)}…` : v
}
