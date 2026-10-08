import { describe, expect, test } from "bun:test"
import { fitView, MIN_FIT, MIN_LEGIBLE, screenTransform } from "./graph-panzoom.ts"

const view = { width: 800, height: 600 }
const scaleOf = (graph: { width: number; height: number }) => screenTransform(fitView({ graph, view }), view.width, view.height).s

describe("fitView", () => {
  test("small graph fits whole, capped zoom", () => expect(scaleOf({ width: 200, height: 100 })).toBeCloseTo(1.25))
  test("tall graph fits the width and starts at the top", () => {
    const graph = { width: 700, height: 3000 }
    const box = fitView({ graph, view })
    expect(scaleOf(graph)).toBeCloseTo(800 / 700)
    expect(box.y).toBe(0)
  })
  test("wide graph fits the width below the legible scale, so it renders as chips", () => {
    const graph = { width: 1600, height: 2400 }
    expect(scaleOf(graph)).toBeCloseTo(0.5)
    expect(scaleOf(graph)).toBeLessThan(MIN_LEGIBLE)
    expect(fitView({ graph, view }).y).toBe(0)
  })
  test("very wide graph stops at the chip floor and centers", () => {
    const graph = { width: 8000, height: 9000 }
    const box = fitView({ graph, view })
    expect(scaleOf(graph)).toBeCloseTo(MIN_FIT)
    expect(box.x + box.w / 2).toBeCloseTo(graph.width / 2)
  })
})
