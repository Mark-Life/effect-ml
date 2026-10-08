export interface Point {
  readonly x: number
  readonly y: number
}

type Segment =
  | { readonly kind: "L"; readonly to: Point }
  | { readonly kind: "C"; readonly c1: Point; readonly c2: Point; readonly to: Point }

/** Uniform cubic B-spline through `points` (same curve as d3 `curveBasis`), as a start point plus segments. */
export const basisSegments = (points: ReadonlyArray<Point>) => {
  const segments: Array<Segment> = []
  const first = points[0]
  if (!first) return { start: { x: 0, y: 0 }, segments }
  if (points.length === 2) return { start: first, segments: [{ kind: "L" as const, to: points[1]! }] }
  let p0 = first
  let p1 = points[1] ?? first
  const curve = (p: Point) => {
    segments.push({
      kind: "C",
      c1: { x: (2 * p0.x + p1.x) / 3, y: (2 * p0.y + p1.y) / 3 },
      c2: { x: (p0.x + 2 * p1.x) / 3, y: (p0.y + 2 * p1.y) / 3 },
      to: { x: (p0.x + 4 * p1.x + p.x) / 6, y: (p0.y + 4 * p1.y + p.y) / 6 }
    })
    p0 = p1
    p1 = p
  }
  if (points.length > 2) segments.push({ kind: "L", to: { x: (5 * p0.x + p1.x) / 6, y: (5 * p0.y + p1.y) / 6 } })
  for (let i = 2; i < points.length; i++) curve(points[i]!)
  if (points.length > 2) {
    curve(p1)
    segments.push({ kind: "L", to: p1 })
  }
  return { start: first, segments }
}

const f = (n: number) => n.toFixed(1)

export const smoothPath = (points: ReadonlyArray<Point>) => {
  const { start, segments } = basisSegments(points)
  return `M${f(start.x)},${f(start.y)}` + segments.map((s) =>
    s.kind === "L"
      ? `L${f(s.to.x)},${f(s.to.y)}`
      : `C${f(s.c1.x)},${f(s.c1.y)} ${f(s.c2.x)},${f(s.c2.y)} ${f(s.to.x)},${f(s.to.y)}`
  ).join("")
}

export interface Polyline {
  readonly xs: Float32Array
  readonly ys: Float32Array
  /** Cumulative arc length at each point. */
  readonly lengths: Float32Array
  readonly total: number
}

const cubic = (a: number, b: number, c: number, d: number, t: number) => {
  const u = 1 - t
  return u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d
}

/** Flattens the smoothed edge into about `samples` points with cumulative arc length. */
export const samplePolyline = (points: ReadonlyArray<Point>, samples = 24): Polyline => {
  const { start, segments } = basisSegments(points)
  const out: Array<Point> = [start]
  const curves = segments.filter((s) => s.kind === "C").length
  const perCurve = Math.max(2, Math.ceil(samples / Math.max(1, curves)))
  let from = start
  for (const s of segments) {
    if (s.kind === "L") out.push(s.to)
    else {
      for (let i = 1; i <= perCurve; i++) {
        const t = i / perCurve
        out.push({ x: cubic(from.x, s.c1.x, s.c2.x, s.to.x, t), y: cubic(from.y, s.c1.y, s.c2.y, s.to.y, t) })
      }
    }
    from = s.to
  }
  const xs = Float32Array.from(out, (p) => p.x)
  const ys = Float32Array.from(out, (p) => p.y)
  const lengths = new Float32Array(out.length)
  for (let i = 1; i < out.length; i++) lengths[i] = lengths[i - 1]! + Math.hypot(xs[i]! - xs[i - 1]!, ys[i]! - ys[i - 1]!)
  return { xs, ys, lengths, total: lengths[out.length - 1] ?? 0 }
}

/** Point at arc length `d` (clamped). Writes into `out` to avoid allocation in the frame loop. */
export const pointAt = (poly: Polyline, d: number, out: { x: number; y: number }) => {
  const { xs, ys, lengths } = poly
  const n = lengths.length
  if (n === 0) return out
  const dist = Math.min(poly.total, Math.max(0, d))
  let lo = 1
  let hi = n - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (lengths[mid]! < dist) lo = mid + 1
    else hi = mid
  }
  const i = Math.max(1, Math.min(lo, n - 1))
  const a = lengths[i - 1] ?? 0
  const seg = (lengths[i] ?? a) - a
  const k = seg > 0 ? (dist - a) / seg : 0
  out.x = xs[i - 1]! + ((xs[i] ?? xs[i - 1]!) - xs[i - 1]!) * k
  out.y = ys[i - 1]! + ((ys[i] ?? ys[i - 1]!) - ys[i - 1]!) * k
  return out
}

interface EdgeRef {
  readonly from: number
  readonly to: number
}

/** Longest-path depth from any source, in topological order. Nodes on a cycle keep depth 0. */
export const nodeDepths = (ids: ReadonlyArray<number>, edges: ReadonlyArray<EdgeRef>) => {
  const indegree = new Map(ids.map((id) => [id, 0]))
  const out = new Map<number, Array<number>>()
  for (const e of edges) {
    indegree.set(e.to, (indegree.get(e.to) ?? 0) + 1)
    out.set(e.from, [...(out.get(e.from) ?? []), e.to])
  }
  const depth = new Map(ids.map((id) => [id, 0]))
  const queue = ids.filter((id) => indegree.get(id) === 0)
  for (let q = 0; q < queue.length; q++) {
    const id = queue[q]!
    for (const next of out.get(id) ?? []) {
      depth.set(next, Math.max(depth.get(next) ?? 0, (depth.get(id) ?? 0) + 1))
      const left = (indegree.get(next) ?? 1) - 1
      indegree.set(next, left)
      if (left === 0) queue.push(next)
    }
  }
  return depth
}

/** Per-edge `[start, end]` of the wave front, normalized by the deepest node. */
export const edgeWindows = (depths: ReadonlyMap<number, number>, edges: ReadonlyArray<EdgeRef>) => {
  const max = Math.max(1, ...depths.values())
  return edges.map((e) => {
    const a = (depths.get(e.from) ?? 0) / max
    const b = (depths.get(e.to) ?? 0) / max
    return b > a ? ([a, b] as const) : ([a, Math.min(1, a + 1 / max)] as const)
  })
}

const reach = (start: number, adjacency: ReadonlyMap<number, ReadonlyArray<number>>) => {
  const seen = new Set([start])
  const stack = [start]
  while (stack.length > 0) {
    for (const n of adjacency.get(stack.pop()!) ?? []) {
      if (seen.has(n)) continue
      seen.add(n)
      stack.push(n)
    }
  }
  return seen
}

/** Ancestors and descendants of `id`, including itself. */
export const activePath = (id: number, edges: ReadonlyArray<EdgeRef>) => {
  const up = new Map<number, Array<number>>()
  const down = new Map<number, Array<number>>()
  for (const e of edges) {
    up.set(e.to, [...(up.get(e.to) ?? []), e.from])
    down.set(e.from, [...(down.get(e.from) ?? []), e.to])
  }
  return new Set([...reach(id, up), ...reach(id, down)])
}
