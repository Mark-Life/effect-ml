import { describe, expect, test } from "bun:test"
import { activePath, edgeWindows, nodeDepths, pointAt, samplePolyline, smoothPath } from "./graph-path.ts"

const pts = [{ x: 0, y: 0 }, { x: 0, y: 10 }, { x: 10, y: 20 }, { x: 10, y: 30 }]

describe("smoothPath", () => {
  test("two points make a line", () => expect(smoothPath([{ x: 0, y: 0 }, { x: 5, y: 5 }])).toBe("M0.0,0.0L5.0,5.0"))
  test("starts and ends on the dagre end points", () => {
    const d = smoothPath(pts)
    expect(d.startsWith("M0.0,0.0")).toBe(true)
    expect(d.endsWith("L10.0,30.0")).toBe(true)
    expect(d).toContain("C")
  })
})

describe("samplePolyline", () => {
  test("arc length grows and ends at the last point", () => {
    const poly = samplePolyline(pts)
    expect(poly.xs.length).toBeGreaterThanOrEqual(20)
    for (let i = 1; i < poly.lengths.length; i++) expect(poly.lengths[i]!).toBeGreaterThanOrEqual(poly.lengths[i - 1]!)
    expect(poly.xs[poly.xs.length - 1]).toBe(10)
    expect(poly.ys[poly.ys.length - 1]).toBe(30)
    expect(poly.total).toBeGreaterThanOrEqual(30)
  })
  test("pointAt clamps to the ends", () => {
    const poly = samplePolyline(pts)
    expect(pointAt(poly, -5, { x: 0, y: 0 })).toEqual({ x: 0, y: 0 })
    const end = pointAt(poly, 1e9, { x: 0, y: 0 })
    expect(end.x).toBeCloseTo(10)
    expect(end.y).toBeCloseTo(30)
  })
})

describe("depths", () => {
  const edges = [{ from: 1, to: 2 }, { from: 2, to: 3 }, { from: 1, to: 3 }, { from: 4, to: 3 }]
  test("longest path from any source", () => {
    const d = nodeDepths([1, 2, 3, 4], edges)
    expect([...d.entries()].sort()).toEqual([[1, 0], [2, 1], [3, 2], [4, 0]])
  })
  test("windows are normalized and ordered", () => {
    const w = edgeWindows(nodeDepths([1, 2, 3, 4], edges), edges)
    expect(w).toEqual([[0, 0.5], [0.5, 1], [0, 1], [0, 1]])
  })
  test("active path holds ancestors and descendants", () => {
    expect([...activePath(2, edges)].sort()).toEqual([1, 2, 3])
  })
})
