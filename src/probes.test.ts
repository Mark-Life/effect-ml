import { describe, expect, test } from "bun:test"
import { downsample, fourierNorms, gridChannel, heatmapProbe, histogram, toMatrix } from "./probes.ts"

const filled = (rows: number, cols: number, f: (r: number, c: number) => number = () => 1) => {
  const data = new Float32Array(rows * cols)
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) data[r * cols + c] = f(r, c)
  return { data, rows, cols }
}

describe("toMatrix", () => {
  test("folds leading dims", () => {
    const m = toMatrix(new Float32Array(24), [2, 3, 4])
    expect([m.rows, m.cols]).toEqual([6, 4])
  })
  test("rank 1 is a row", () => {
    const m = toMatrix(new Float32Array(5), [5])
    expect([m.rows, m.cols]).toEqual([1, 5])
  })
})

describe("downsample", () => {
  test.each([
    [113, 113],
    [512, 256],
    [1000, 250]
  ])("%i -> %i", (n, expected) => {
    const m = downsample(filled(n, 7))
    expect(m.rows).toBe(expected)
    expect(m.cols).toBe(7)
  })

  test("block mean, partial edge blocks average only their cells", () => {
    // 5 rows, factor 2 -> rows [0,1], [2,3], [4]
    const m = downsample(filled(5, 1, (r) => r), 3)
    expect([...m.data]).toEqual([0.5, 2.5, 4])
  })
})

describe("histogram", () => {
  test("counts sum to finite count", () => {
    const data = Float32Array.from([0, 1, 2, 3, Number.NaN, Infinity, -1, 0.5])
    const h = histogram(data, 4)
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(6)
    expect(h.nonFinite).toBe(2)
    expect([h.min, h.max]).toEqual([-1, 3])
    expect(h.counts[3]).toBe(2)
  })
  test("constant input lands in bin 0", () => {
    const h = histogram(Float32Array.from([2, 2, 2]), 8)
    expect(h.counts[0]).toBe(3)
  })
})

describe("gridChannel", () => {
  test("index math: example a*p+b, channel within rest", () => {
    const p = 3
    const rest = [2, 2]
    const data = new Float32Array(p * p * 4)
    for (let i = 0; i < p * p; i++) for (let c = 0; c < 4; c++) data[i * 4 + c] = i * 10 + c
    const m = gridChannel({ data, shape: [p * p, ...rest], channel: 3, p })
    expect([m.rows, m.cols]).toEqual([3, 3])
    // a = 2, b = 1 -> example 7
    expect(m.data[2 * p + 1]).toBe(73)
  })
  test("rejects wrong leading dim", () => {
    expect(() => gridChannel({ data: new Float32Array(8), shape: [8], channel: 0, p: 3 })).toThrow()
  })
})

describe("fourierNorms", () => {
  test("pure cosine peaks at its frequency", () => {
    const p = 113
    const k = 17
    const W = filled(p, 2, (t, c) => Math.cos((2 * Math.PI * k * t) / p + c))
    const F = fourierNorms(W, p)
    expect(F.rows).toBe(57)
    for (let c = 0; c < 2; c++) {
      const column = Array.from({ length: F.rows }, (_, r) => F.data[r * 2 + c]!)
      expect(column.indexOf(Math.max(...column))).toBe(k)
      expect(column[k]).toBeCloseTo(p / 2, 1)
    }
  })
})

describe("heatmapProbe mask", () => {
  test("appends the mask plane after the values", () => {
    const { header, payload } = heatmapProbe({
      name: "correct",
      step: 1,
      sourceShape: [4],
      matrix: filled(2, 2, (r, c) => r + c),
      scale: "sequential",
      mask: { dimmed: "train", shown: "test", data: Float32Array.of(1, 0, 0, 1) }
    })
    expect(header.mask).toEqual({ dimmed: "train", shown: "test" })
    expect([...payload]).toEqual([0, 1, 1, 2, 1, 0, 0, 1])
  })
  test("downsampled blocks are masked by majority", () => {
    const { header, payload } = heatmapProbe({
      name: "m",
      step: 1,
      sourceShape: [512, 1],
      matrix: filled(512, 1),
      scale: "sequential",
      mask: { dimmed: "train", shown: "test", data: Float32Array.from({ length: 512 }, (_, i) => (i < 256 ? 1 : 0)) }
    })
    expect(header.rows).toBe(256)
    expect(payload.subarray(256, 384).every((v) => v === 1)).toBe(true)
    expect(payload.subarray(384).every((v) => v === 0)).toBe(true)
  })
})
