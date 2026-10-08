import type { HeatmapHeader } from "../protocol.ts"

const EPS = 1e-12

/** Colormap range for a heatmap. Widens degenerate ranges so a constant map lands on a meaningful color. */
export const heatmapRange = ({ min, max, scale }: Pick<HeatmapHeader, "min" | "max" | "scale">) => {
  if (scale === "diverging") {
    const absMax = Math.max(Math.abs(min), Math.abs(max))
    return absMax === 0 ? { lo: -1, hi: 1, constant: true } : { lo: -absMax, hi: absMax, constant: false }
  }
  if (max - min > EPS * Math.max(1, Math.abs(max))) return { lo: min, hi: max, constant: false }
  const lo = Math.min(min, 0)
  const hi = Math.max(max, 0)
  return { lo, hi: hi > lo ? hi : lo + 1, constant: true }
}

/** True when every finite value equals `min` or `max` and the two differ: a mask or boolean map. */
export const isTwoLevel = (payload: Float32Array, { min, max }: Pick<HeatmapHeader, "min" | "max">) => {
  if (!(max > min)) return false
  for (const v of payload) if (Number.isFinite(v) && v !== min && v !== max) return false
  return true
}

/**
 * Splits a two-level map into its low and high shares and lights the smaller group, so the rare cells stand
 * out whichever level they are. Ties go to `prefer`. Cells set in `mask` are left out.
 */
export const twoLevelSplit = (
  { payload, lo, hi, prefer = "high", mask }: {
    payload: Float32Array
    lo: number
    hi: number
    prefer?: "low" | "high"
    mask?: Float32Array | undefined
  }
) => {
  let low = 0
  let high = 0
  const mid = (lo + hi) / 2
  for (let i = 0; i < payload.length; i++) {
    const v = payload[i]!
    if (!Number.isFinite(v) || (mask && mask[i]! >= 0.5)) continue
    if (v >= mid) high++
    else low++
  }
  const total = Math.max(1, low + high)
  const lowShare = low / total
  const highShare = high / total
  const lit: "low" | "high" = lowShare === highShare ? prefer : lowShare < highShare ? "low" : "high"
  return { lowShare, highShare, lit }
}

/** Share of cells clipped at each end of a continuous map, so a few outliers cannot wash out the rest. */
const CLIP = 0.005
/** Log ramp kicks in when the median sits this low in the clipped range: most cells would be near-black. */
const SKEW = 0.12
/** Probabilities are clamped this far from 0 and 1 before log-odds. */
export const LOGIT_EPS = 1e-6

/** Value transform the shader applies before the linear `lo..hi` mapping (`log` acts after it, on `t`). */
export const RAMPS = { linear: 0, log: 1, logit: 2 } as const
export type Ramp = keyof typeof RAMPS

export const logit = (p: number) => {
  const q = Math.min(1 - LOGIT_EPS, Math.max(LOGIT_EPS, p))
  return Math.log(q / (1 - q))
}

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

const quantile = (sorted: Float32Array, q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]!

const finiteSorted = (payload: Float32Array, f: (v: number) => number) => {
  const out = new Float32Array(payload.length)
  let n = 0
  for (const v of payload) if (Number.isFinite(v)) out[n++] = f(v)
  return out.subarray(0, n).sort()
}

const identity = (v: number) => v

/**
 * Colormap range from the 0.5 and 99.5 percentiles instead of min and max. Probability maps are ranged in
 * log-odds (`lo`/`hi` are logits). A sequential map whose mass sits at the bottom of the range (sparse
 * activations) gets the `log` ramp.
 */
export const robustRange = (
  payload: Float32Array,
  header: Pick<HeatmapHeader, "min" | "max" | "scale">
): { lo: number; hi: number; constant: boolean; ramp: Ramp } => {
  const plain = heatmapRange(header)
  if (plain.constant) return { ...plain, ramp: "linear" }
  if (header.scale === "diverging") {
    const sorted = finiteSorted(payload, Math.abs)
    const absMax = sorted.length === 0 ? 0 : quantile(sorted, 1 - CLIP)
    if (!(absMax > 0)) return { ...plain, ramp: "linear" }
    return { lo: -absMax, hi: absMax, constant: false, ramp: "linear" }
  }
  const probability = header.scale === "probability"
  const sorted = finiteSorted(payload, probability ? logit : identity)
  if (sorted.length === 0) return { ...plain, ramp: "linear" }
  const lo = quantile(sorted, CLIP)
  const hi = quantile(sorted, 1 - CLIP)
  if (!(hi - lo > EPS * Math.max(1, Math.abs(hi)))) {
    return probability ? { lo: logit(header.min), hi: logit(header.max), constant: false, ramp: "logit" } : { ...plain, ramp: "linear" }
  }
  if (probability) return { lo, hi, constant: false, ramp: "logit" }
  const median = quantile(sorted, 0.5)
  return { lo, hi, constant: false, ramp: (median - lo) / (hi - lo) < SKEW ? "log" : "linear" }
}
