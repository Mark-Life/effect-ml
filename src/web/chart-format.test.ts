import { describe, expect, test } from "bun:test"
import { evalDelta, plotHeight } from "./chart-format.ts"

describe("plotHeight", () => {
  test("clamps", () => {
    expect(plotHeight(300)).toBe(180)
    expect(plotHeight(700)).toBe(245)
    expect(plotHeight(2000)).toBe(320)
  })
})

describe("evalDelta", () => {
  test("first eval has no delta", () => expect(evalDelta({ now: 1, before: undefined, kind: "loss" }).text).toBe("–"))
  test("loss drop is relative and better", () =>
    expect(evalDelta({ now: 0.5, before: 1, kind: "loss" })).toEqual({ text: "Δ −50.0%", trend: "better" }))
  test("accuracy rise is absolute and better", () =>
    expect(evalDelta({ now: 0.75, before: 0.5, kind: "acc" })).toEqual({ text: "Δ +0.250", trend: "better" }))
  test("loss rise is worse", () => expect(evalDelta({ now: 1.2, before: 1, kind: "loss" }).trend).toBe("worse"))
  test("tiny loss change is flat", () =>
    expect(evalDelta({ now: 1.0004, before: 1, kind: "loss" })).toEqual({ text: "Δ —", trend: "flat" }))
  test("accuracy change below shown precision is flat", () =>
    expect(evalDelta({ now: 0.5002, before: 0.5, kind: "acc" })).toEqual({ text: "Δ —", trend: "flat" }))
})
