import { describe, expect, test } from "bun:test"
import { heatmapRange, isTwoLevel, logit, robustRange, twoLevelSplit } from "./heatmap-range.ts"

const seq = (min: number, max: number) => heatmapRange({ min, max, scale: "sequential" })

describe("heatmapRange", () => {
  test("all ones map to the top color", () => expect(seq(1, 1)).toEqual({ lo: 0, hi: 1, constant: true }))
  test("all zeros map to the bottom color", () => expect(seq(0, 0)).toEqual({ lo: 0, hi: 1, constant: true }))
  test("negative constant", () => expect(seq(-2, -2)).toEqual({ lo: -2, hi: 0, constant: true }))
  test("normal range unchanged", () => expect(seq(0.1, 3)).toEqual({ lo: 0.1, hi: 3, constant: false }))
  test("diverging zero", () =>
    expect(heatmapRange({ min: 0, max: 0, scale: "diverging" })).toEqual({ lo: -1, hi: 1, constant: true }))
  test("diverging symmetric", () =>
    expect(heatmapRange({ min: -0.5, max: 2, scale: "diverging" })).toEqual({ lo: -2, hi: 2, constant: false }))
})

describe("isTwoLevel", () => {
  const range = { min: 0, max: 1 }
  test("boolean map", () => expect(isTwoLevel(new Float32Array([0, 1, 1, 0]), range)).toBe(true))
  test("ignores non-finite cells", () => expect(isTwoLevel(new Float32Array([0, Number.NaN, 1]), range)).toBe(true))
  test("third value", () => expect(isTwoLevel(new Float32Array([0, 0.5, 1]), range)).toBe(false))
  test("constant map", () => expect(isTwoLevel(new Float32Array([1, 1]), { min: 1, max: 1 })).toBe(false))
})

describe("twoLevelSplit", () => {
  const split = (values: ReadonlyArray<number>, prefer?: "low" | "high") =>
    twoLevelSplit({ payload: new Float32Array(values), lo: 0, hi: 1, ...(prefer ? { prefer } : {}) })
  test("lights the rare low cells", () => expect(split([0, 1, 1, 1])).toEqual({ lowShare: 0.25, highShare: 0.75, lit: "low" }))
  test("lights the rare high cells", () => expect(split([0, 0, 0, 1]).lit).toBe("high"))
  test("all high lights nothing", () => expect(split([1, 1])).toEqual({ lowShare: 0, highShare: 1, lit: "low" }))
  test("tie follows prefer", () => expect(split([0, 1], "low").lit).toBe("low"))
  test("skips non-finite", () => expect(split([0, Number.NaN, 1, 1]).lowShare).toBeCloseTo(1 / 3))
  test("skips masked cells", () => {
    const r = twoLevelSplit({ payload: new Float32Array([1, 1, 1, 0]), lo: 0, hi: 1, mask: new Float32Array([1, 1, 0, 0]) })
    expect(r).toEqual({ lowShare: 0.5, highShare: 0.5, lit: "high" })
  })
})

describe("robustRange", () => {
  const ramp = Float32Array.from({ length: 1000 }, (_, i) => i / 999)
  test("clips a lone outlier", () => {
    const data = Float32Array.from(ramp)
    data[500] = 100
    const r = robustRange(data, { min: 0, max: 100, scale: "sequential" })
    expect(r.hi).toBeLessThan(1.01)
    expect(r.ramp).toBe("linear")
  })
  test("sparse map goes log", () => {
    const data = Float32Array.from({ length: 1000 }, (_, i) => (i % 20 === 0 ? 1 : 0.01))
    expect(robustRange(data, { min: 0.01, max: 1, scale: "sequential" }).ramp).toBe("log")
  })
  test("diverging stays symmetric", () => {
    const r = robustRange(Float32Array.from(ramp, (v) => v * 2 - 1), { min: -1, max: 1, scale: "diverging" })
    expect(r.lo).toBe(-r.hi)
  })
  test("probability map ranges in log-odds", () => {
    const data = Float32Array.from({ length: 1000 }, (_, i) => (i < 900 ? 0.999 : 0.3))
    const r = robustRange(data, { min: 0.3, max: 0.999, scale: "probability" })
    expect(r.ramp).toBe("logit")
    expect(r.hi).toBeCloseTo(logit(0.999), 3)
    expect(r.lo).toBeCloseTo(logit(0.3), 3)
  })
  test("constant map falls back", () =>
    expect(robustRange(new Float32Array([1, 1]), { min: 1, max: 1, scale: "sequential" }).constant).toBe(true))
})
