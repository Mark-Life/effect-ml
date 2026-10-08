import { makeGlowSprite } from "./glow.ts"
import { type Polyline, pointAt } from "./graph-path.ts"
import { screenTransform, type ViewBox } from "./graph-panzoom.ts"
import { theme } from "./theme.ts"

const WAVE_MS = 1400
const MAX_WAVES = 2
const NEXT_WAVE_AT = 0.5
/** Particles drawn per wave at once, longest edges first, so a wave reads as one signal. */
const MAX_PER_WAVE = 3
const HALO_PX = 18
const CORE_R = 1.25
const TRAIL_PX = 36
const TRAIL_SEGMENTS = 8
/** Halo fades over the last part of the travel, while the particle settles on the node. */
const REST_FADE = 0.4

export interface PulseEdge {
  readonly poly: Polyline
  readonly window: readonly [number, number]
  /** Train graph: the edge feeds a backward or optimizer node. */
  readonly accent: boolean
  /** The edge enters the loss node; it keeps a static accent mark when motion is off. */
  readonly lossEntry: boolean
}

export type PulseMode = "forward" | "train"

interface Wave {
  readonly t0: number
  readonly reverse: boolean
  spawned: boolean
}

const easeOut = (t: number) => 1 - (1 - t) ** 3

/**
 * Canvas overlay over the graph SVG. Waves start from incoming metrics and sweep the DAG in depth order;
 * each edge carries one glowing particle while the wave front is inside its depth window. `onWave` gets the step
 * that started a wave; the step label lives outside the canvas so it never covers node labels.
 */
export const makePulses = (
  { host, viewBox, onWave }: { host: HTMLElement; viewBox: () => ViewBox; onWave: (step: number) => void }
) => {
  const canvas = document.createElement("canvas")
  canvas.className = "graph-pulses"
  host.append(canvas)
  const ctx = canvas.getContext("2d")
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  const sprite = makeGlowSprite(HALO_PX, dpr)
  const head = { x: 0, y: 0 }
  const tmp = { x: 0, y: 0 }
  let width = 0
  let height = 0
  let edges: ReadonlyArray<PulseEdge> = []
  let ranked: ReadonlyArray<PulseEdge> = []
  let mode: PulseMode = "train"
  let waves: Array<Wave> = []
  let enabled = false
  let training = false
  let raf = 0

  const drawStatic = (s: number, tx: number, ty: number) => {
    if (!ctx) return
    ctx.strokeStyle = theme.accent
    ctx.lineWidth = 1.5
    ctx.globalAlpha = 1
    for (const e of edges) {
      if (!e.lossEntry) continue
      ctx.beginPath()
      for (let k = 0; k <= 4; k++) {
        pointAt(e.poly, e.poly.total - (k * 3) / s, tmp)
        if (k === 0) ctx.moveTo(tmp.x * s + tx, tmp.y * s + ty)
        else ctx.lineTo(tmp.x * s + tx, tmp.y * s + ty)
      }
      ctx.stroke()
    }
  }

  const drawParticle = (
    { e, wave, local, s, tx, ty }: { e: PulseEdge; wave: Wave; local: number; s: number; tx: number; ty: number }
  ) => {
    if (!ctx) return
    const eased = easeOut(local)
    const dist = wave.reverse ? e.poly.total * (1 - eased) : e.poly.total * eased
    const back = wave.reverse ? 1 : -1
    const alpha = Math.min(1, local / 0.06, (1 - local) / 0.1)
    if (alpha <= 0) return
    pointAt(e.poly, dist, head)
    const hx = head.x * s + tx
    const hy = head.y * s + ty
    const backward = wave.reverse || (mode === "train" && e.accent)
    ctx.globalCompositeOperation = "source-over"
    ctx.lineWidth = 1
    ctx.strokeStyle = theme.accent
    let px = hx
    let py = hy
    for (let k = 1; k <= TRAIL_SEGMENTS; k++) {
      pointAt(e.poly, Math.max(0, Math.min(e.poly.total, dist + (back * k * TRAIL_PX) / TRAIL_SEGMENTS / s)), tmp)
      const x = tmp.x * s + tx
      const y = tmp.y * s + ty
      ctx.globalAlpha = alpha * 0.7 * (1 - (k - 1) / TRAIL_SEGMENTS)
      ctx.beginPath()
      ctx.moveTo(px, py)
      ctx.lineTo(x, y)
      ctx.stroke()
      px = x
      py = y
    }
    const halo = alpha * Math.min(1, (1 - local) / REST_FADE)
    ctx.globalCompositeOperation = "lighter"
    ctx.globalAlpha = halo * (backward ? 0.7 : 1)
    ctx.drawImage(sprite, hx - HALO_PX / 2, hy - HALO_PX / 2, HALO_PX, HALO_PX)
    ctx.globalCompositeOperation = "source-over"
    ctx.globalAlpha = alpha
    ctx.fillStyle = theme.hot
    ctx.beginPath()
    ctx.arc(hx, hy, CORE_R, 0, Math.PI * 2)
    ctx.fill()
  }

  const draw = (now: number) => {
    if (!ctx || width === 0) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.globalCompositeOperation = "source-over"
    ctx.clearRect(0, 0, width, height)
    const { s, tx, ty } = screenTransform(viewBox(), width, height)
    if (!enabled) return drawStatic(s, tx, ty)
    ctx.lineWidth = 1
    for (const wave of waves) {
      const f = (now - wave.t0) / WAVE_MS
      let drawn = 0
      for (const e of ranked) {
        if (drawn >= MAX_PER_WAVE) break
        const [a, b] = wave.reverse ? [1 - e.window[1], 1 - e.window[0]] : e.window
        if (f < a || f > b) continue
        drawn++
        drawParticle({ e, wave, local: (f - a) / (b - a), s, tx, ty })
      }
    }
  }

  const step = (now: number) => {
    raf = 0
    for (const wave of waves) {
      if (mode !== "forward" || wave.reverse || wave.spawned || (now - wave.t0) / WAVE_MS < 1) continue
      wave.spawned = true
      waves.push({ t0: now, reverse: true, spawned: true })
    }
    waves = waves.filter((w) => (now - w.t0) / WAVE_MS <= 1.02)
    draw(now)
    if (enabled && waves.length > 0) raf = requestAnimationFrame(step)
  }

  const kick = () => {
    if (raf === 0) raf = requestAnimationFrame(step)
  }

  const redraw = () => {
    if (raf === 0) draw(performance.now())
  }

  new ResizeObserver(([entry]) => {
    if (!entry) return
    width = entry.contentRect.width
    height = entry.contentRect.height
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    redraw()
  }).observe(canvas)

  return {
    setLayout: (next: { edges: ReadonlyArray<PulseEdge>; mode: PulseMode }) => {
      edges = next.edges
      ranked = [...edges].sort((x, y) => Number(y.lossEntry) - Number(x.lossEntry) || y.poly.total - x.poly.total)
      mode = next.mode
      waves = []
      redraw()
    },
    setEnabled: (on: boolean) => {
      enabled = on
      if (!on) waves = []
      redraw()
    },
    setTraining: (on: boolean) => void (training = on),
    onMetrics: (stepNumber: number) => {
      if (!enabled || !training || edges.length === 0 || waves.length >= MAX_WAVES) return
      const now = performance.now()
      const newest = waves[waves.length - 1]
      if (newest && (now - newest.t0) / WAVE_MS <= NEXT_WAVE_AT) return
      waves.push({ t0: now, reverse: false, spawned: false })
      onWave(stepNumber)
      kick()
    },
    invalidate: redraw,
    /** True while waves are in flight. */
    busy: () => waves.length > 0
  }
}

export type Pulses = ReturnType<typeof makePulses>
