import { describe, expect, test } from "bun:test"
import { buildLut, buildTwoTone, hexToOklab, LUT_SIZE } from "./colormap-lut.ts"
import { divergingStops, emberStops, hexToRgb, theme } from "./theme.ts"

const texel = (lut: Uint8Array, i: number) => [...lut.subarray(i * 4, i * 4 + 3)]
const lightness = (lut: Uint8Array, i: number) => {
  const [r, g, b] = texel(lut, i)
  const hex = `#${[r, g, b].map((c) => c!.toString(16).padStart(2, "0")).join("")}`
  return hexToOklab(hex)[0]
}

describe("buildLut", () => {
  test("endpoints are exact", () => {
    const ember = buildLut(emberStops)
    expect(texel(ember, 0)).toEqual([...hexToRgb(theme.bg)])
    expect(texel(ember, LUT_SIZE - 1)).toEqual([...hexToRgb(theme.heatTop)])
    const diverging = buildLut(divergingStops)
    expect(texel(diverging, LUT_SIZE - 1)).toEqual([...hexToRgb(theme.heatTop)])
  })

  test("ember lightness is monotonic", () => {
    const ember = buildLut(emberStops)
    for (let i = 1; i < LUT_SIZE; i++) expect(lightness(ember, i)).toBeGreaterThanOrEqual(lightness(ember, i - 1) - 1e-3)
  })

  test("diverging is darkest at zero", () => {
    const d = buildLut(divergingStops)
    const mid = Math.round((LUT_SIZE - 1) / 2)
    expect(lightness(d, mid)).toBeLessThan(lightness(d, 0))
    expect(lightness(d, mid)).toBeLessThan(lightness(d, LUT_SIZE - 1))
  })

  test("diverging sides are equally bright", () => {
    const d = buildLut(divergingStops)
    for (let i = 0; i < LUT_SIZE / 2; i += 8) {
      expect(Math.abs(lightness(d, i) - lightness(d, LUT_SIZE - 1 - i))).toBeLessThan(0.03)
    }
  })

  test("ember keeps the accent out of the lower 70%", () => {
    const ember = buildLut(emberStops)
    expect(lightness(ember, Math.round(0.7 * (LUT_SIZE - 1)))).toBeLessThan(hexToOklab(theme.accent)[0] - 0.2)
  })

  test("two-tone splits at the middle", () => {
    const lut = buildTwoTone({ lo: theme.faint, hi: theme.accent })
    expect(texel(lut, 0)).toEqual([...hexToRgb(theme.faint)])
    expect(texel(lut, LUT_SIZE / 2 - 1)).toEqual([...hexToRgb(theme.faint)])
    expect(texel(lut, LUT_SIZE / 2)).toEqual([...hexToRgb(theme.accent)])
  })
})
