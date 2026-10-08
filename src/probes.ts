import type { HeatmapHeader, HistogramHeader } from "./protocol.ts"

export interface Matrix {
  readonly data: Float32Array
  readonly rows: number
  readonly cols: number
}

/** Seeded PRNG returning floats in [0, 1). */
export const mulberry32 = (seed: number) => {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const product = (xs: ReadonlyArray<number>) => xs.reduce((a, b) => a * b, 1)

/** Leading dims fold into rows, last dim is cols. Rank 0/1 becomes `1 x n`. */
export const toMatrix = (data: Float32Array, shape: ReadonlyArray<number>): Matrix => {
  const cols = shape.length === 0 ? 1 : shape[shape.length - 1]!
  return { data, rows: shape.length <= 1 ? 1 : product(shape.slice(0, -1)), cols }
}

/** Block mean so each side is at most `max`. Edge blocks average only the cells they hold. */
export const downsample = (m: Matrix, max = 256): Matrix => {
  const fr = Math.ceil(m.rows / max)
  const fc = Math.ceil(m.cols / max)
  if (fr === 1 && fc === 1) return m
  const rows = Math.ceil(m.rows / fr)
  const cols = Math.ceil(m.cols / fc)
  const sums = new Float64Array(rows * cols)
  const counts = new Uint32Array(rows * cols)
  for (let r = 0; r < m.rows; r++) {
    const base = Math.floor(r / fr) * cols
    for (let c = 0; c < m.cols; c++) {
      const v = m.data[r * m.cols + c]!
      if (!Number.isFinite(v)) continue
      const i = base + Math.floor(c / fc)
      sums[i] = sums[i]! + v
      counts[i] = counts[i]! + 1
    }
  }
  const data = new Float32Array(rows * cols)
  for (let i = 0; i < data.length; i++) data[i] = counts[i] === 0 ? Number.NaN : sums[i]! / counts[i]!
  return { data, rows, cols }
}

/**
 * For `[N, ...rest]` with `N = p*p`, takes flat channel `channel` of `rest` per example.
 * Result is `p x p` with rows = a, cols = b (example index `a*p + b`).
 */
export const gridChannel = (
  { data, shape, channel, p }: { data: Float32Array; shape: ReadonlyArray<number>; channel: number; p: number }
): Matrix => {
  const stride = product(shape.slice(1))
  if (shape[0] !== p * p) throw new Error(`gridChannel: expected leading dim ${p * p}, got ${shape[0]}`)
  if (channel < 0 || channel >= stride) throw new Error(`gridChannel: channel ${channel} out of range ${stride}`)
  const out = new Float32Array(p * p)
  for (let i = 0; i < p * p; i++) out[i] = data[i * stride + channel]!
  return { data: out, rows: p, cols: p }
}

export interface Histogram {
  readonly counts: Uint32Array
  readonly min: number
  readonly max: number
  readonly nonFinite: number
}

export const histogram = (data: Float32Array, bins = 64): Histogram => {
  let min = Infinity
  let max = -Infinity
  let nonFinite = 0
  for (const v of data) {
    if (!Number.isFinite(v)) nonFinite++
    else {
      if (v < min) min = v
      if (v > max) max = v
    }
  }
  const counts = new Uint32Array(bins)
  if (nonFinite === data.length) return { counts, min: 0, max: 0, nonFinite }
  const width = (max - min) / bins
  for (const v of data) {
    if (!Number.isFinite(v)) continue
    const bin = width === 0 ? 0 : Math.min(bins - 1, Math.floor((v - min) / width))
    counts[bin] = counts[bin]! + 1
  }
  return { counts, min, max, nonFinite }
}

/** `|DFT_k|` over the token axis per column: `W [p, d]` -> `[floor(p/2)+1, d]`. */
export const fourierNorms = (W: Matrix, p: number): Matrix => {
  const d = W.cols
  const rows = Math.floor(p / 2) + 1
  const out = new Float32Array(rows * d)
  const cos = new Float64Array(p)
  const sin = new Float64Array(p)
  for (let k = 0; k < rows; k++) {
    for (let t = 0; t < p; t++) {
      const angle = (2 * Math.PI * k * t) / p
      cos[t] = Math.cos(angle)
      sin[t] = Math.sin(angle)
    }
    for (let c = 0; c < d; c++) {
      let re = 0
      let im = 0
      for (let t = 0; t < p; t++) {
        const v = W.data[t * d + c]!
        re += v * cos[t]!
        im -= v * sin[t]!
      }
      out[k * d + c] = Math.hypot(re, im)
    }
  }
  return { data: out, rows, cols: d }
}

export const finiteRange = (data: Float32Array) => {
  let min = Infinity
  let max = -Infinity
  for (const v of data) {
    if (Number.isFinite(v)) {
      if (v < min) min = v
      if (v > max) max = v
    }
  }
  return min <= max ? { min, max } : { min: 0, max: 0 }
}

export interface HeatmapInput {
  readonly name: string
  readonly step: number
  readonly sourceShape: ReadonlyArray<number>
  readonly matrix: Matrix
  readonly scale: HeatmapHeader["scale"]
  readonly axes?: HeatmapHeader["axes"]
  readonly highlight?: HeatmapHeader["highlight"]
  readonly levels?: HeatmapHeader["levels"]
  /** Same shape as `matrix`; 1 marks cells the viewer draws dimmed. A downsampled block is masked when most of it is. */
  readonly mask?: NonNullable<HeatmapHeader["mask"]> & { readonly data: Float32Array }
}

/** Downsamples and builds the wire header plus payload (values, then the mask plane when there is one). */
export const heatmapProbe = (
  { name, step, sourceShape, matrix, scale, axes, highlight, levels, mask }: HeatmapInput
) => {
  const m = downsample(matrix)
  const maskPlane = mask
    ? downsample({ data: mask.data, rows: matrix.rows, cols: matrix.cols }).data.map((v) => (v >= 0.5 ? 1 : 0))
    : undefined
  const header: HeatmapHeader = {
    _tag: "Heatmap",
    name,
    step,
    sourceShape,
    rows: m.rows,
    cols: m.cols,
    ...finiteRange(m.data),
    scale,
    ...(axes ? { axes } : {}),
    ...(highlight ? { highlight } : {}),
    ...(levels ? { levels } : {}),
    ...(mask ? { mask: { dimmed: mask.dimmed, shown: mask.shown } } : {})
  }
  if (!maskPlane) return { header, payload: m.data }
  const payload = new Float32Array(m.data.length * 2)
  payload.set(m.data)
  payload.set(maskPlane, m.data.length)
  return { header, payload }
}

export const histogramProbe = (
  { name, step, sourceShape, data, bins = 64 }: {
    name: string
    step: number
    sourceShape: ReadonlyArray<number>
    data: Float32Array
    bins?: number
  }
) => {
  const h = histogram(data, bins)
  const header: HistogramHeader = {
    _tag: "Histogram",
    name,
    step,
    sourceShape,
    bins,
    min: h.min,
    max: h.max,
    nonFinite: h.nonFinite
  }
  return { header, payload: h.counts }
}
