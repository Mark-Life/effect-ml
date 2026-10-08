import { hexToRgb, type Stop } from "./theme.ts"

export const LUT_SIZE = 256

const toLinear = (c: number) => {
  const v = c / 255
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
}

const fromLinear = (v: number) => {
  const c = v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055
  return Math.round(Math.min(1, Math.max(0, c)) * 255)
}

export const hexToOklab = (hex: string) => {
  const [r8, g8, b8] = hexToRgb(hex)
  const r = toLinear(r8)
  const g = toLinear(g8)
  const b = toLinear(b8)
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s
  ] as const
}

const oklabToRgb = ([L, a, b]: readonly [number, number, number]) => {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
  return [
    fromLinear(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    fromLinear(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    fromLinear(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)
  ] as const
}

/** Samples `stops` (sorted by `at`) into an RGBA8 LUT, interpolating in OKLab. */
export const buildLut = (stops: ReadonlyArray<Stop>) => {
  const first = stops[0]!.at
  const span = stops[stops.length - 1]!.at - first
  const labs = stops.map((s) => ({ at: (s.at - first) / span, lab: hexToOklab(s.color) }))
  const out = new Uint8Array(LUT_SIZE * 4)
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1)
    const hi = Math.max(1, labs.findIndex((s) => s.at >= t))
    const a = labs[hi - 1]!
    const b = labs[hi]!
    const k = b.at === a.at ? 0 : (t - a.at) / (b.at - a.at)
    const at = (c: 0 | 1 | 2) => a.lab[c] + (b.lab[c] - a.lab[c]) * k
    out.set([...oklabToRgb([at(0), at(1), at(2)]), 255], i * 4)
  }
  return out
}

/** Hard split LUT: the lower half is `lo`, the upper half `hi`. */
export const buildTwoTone = ({ lo, hi }: { lo: string; hi: string }) => {
  const out = new Uint8Array(LUT_SIZE * 4)
  for (let i = 0; i < LUT_SIZE; i++) out.set([...hexToRgb(i < LUT_SIZE / 2 ? lo : hi), 255], i * 4)
  return out
}

/** LUT rows stacked in one buffer; the row index is the shader `mode`. */
export const buildLutAtlas = (rows: ReadonlyArray<Uint8Array>) => {
  const atlas = new Uint8Array(LUT_SIZE * 4 * rows.length)
  rows.forEach((row, r) => atlas.set(row, r * LUT_SIZE * 4))
  return atlas
}
