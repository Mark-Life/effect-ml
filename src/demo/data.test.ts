import { describe, expect, test } from "bun:test"
import { modularAddition } from "./data.ts"

describe("modularAddition", () => {
  test("30% train split of p*p pairs", () => {
    const split = modularAddition({ p: 113, trainFrac: 0.3 })
    expect(split.n).toBe(12769)
    expect(split.trainIndices.length).toBe(3831)
    expect(split.trainMask.reduce((a, b) => a + b, 0)).toBe(3831)
    expect(new Set(split.trainIndices).size).toBe(3831)
  })
  test("deterministic per seed", () => {
    const a = modularAddition({ p: 113, trainFrac: 0.3 })
    const b = modularAddition({ p: 113, trainFrac: 0.3 })
    const c = modularAddition({ p: 113, trainFrac: 0.3, seed: 1 })
    expect(a.trainIndices).toEqual(b.trainIndices)
    expect(a.trainIndices).not.toEqual(c.trainIndices)
  })
})
