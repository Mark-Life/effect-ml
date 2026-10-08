import { mixHex, prefersReducedMotion, theme } from "../theme.ts"

const LEVELS = 16
const SPIN = 0.08
const TILT = 0.35
const EASE_MS = 600
const BEAT_MS = 120
const IDLE_FPS = 15

interface SphereOptions {
  readonly canvas: HTMLCanvasElement
  /** CSS px, square. */
  readonly size: number
  readonly dpr: number
  readonly fps: number
  /** 0: color is fg-dim at zero heat; 1: always accent. */
  readonly tint: number
  /** Sphere radius as a fraction of the canvas half size. */
  readonly radius: number
  /** Number of dots. */
  readonly count: number
  /** Alpha of the farthest dots; the nearest are opaque. */
  readonly floor: number
  /** How far untrained dots scatter, in sphere radii. */
  readonly scatter: number
  /** Radial rays the scattered dots gather on; 0 scatters them evenly. */
  readonly spokes?: number
}

const SPOKE_SHARE = 0.55
const SPOKE_WIDTH = 0.035

/** Screen-space ray layout: angle, length and ray membership per dot; density falls off along each ray. */
const rays = (N: number, spokes: number) => {
  const angle = new Float32Array(N)
  const reach = new Float32Array(N)
  const on = new Uint8Array(N)
  if (spokes === 0) return { angle, reach, on }
  const lengths = Array.from({ length: spokes }, (_, k) => 0.55 + 0.45 * hash(k + 7.31))
  const offsets = Array.from({ length: spokes }, (_, k) => (hash(k + 3.17) - 0.5) * 0.35)
  for (let i = 0; i < N; i++) {
    if (hash(i + 2 * N) > SPOKE_SHARE) continue
    const k = Math.floor(hash(i + 3 * N) * spokes)
    const spread = (hash(i + 4 * N) - 0.5) * 2 * SPOKE_WIDTH
    on[i] = 1
    angle[i] = ((k + 0.5) / spokes) * Math.PI * 2 + offsets[k]! + spread
    reach[i] = hash(i + 5 * N) ** 2.2 * lengths[k]!
  }
  return { angle, reach, on }
}

const hash = (i: number) => {
  const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453
  return x - Math.floor(x)
}

const fibonacci = (N: number) => {
  const golden = Math.PI * (3 - Math.sqrt(5))
  const unit = new Float32Array(N * 3)
  const jitter = new Float32Array(N)
  const phase = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    const y = 1 - (2 * (i + 0.5)) / N
    const r = Math.sqrt(1 - y * y)
    const a = golden * i
    unit.set([Math.cos(a) * r, y, Math.sin(a) * r], i * 3)
    jitter[i] = hash(i) ** 2
    phase[i] = hash(i + N) * Math.PI * 2
  }
  return { unit, jitter, phase }
}

/**
 * Rotating dotted sphere driven by the run: `order` (test accuracy) pulls a scattered burst onto the sphere,
 * `heat` (loss) warms the color, `beat` flashes on each pulse wave. Dots are binned by depth into a few
 * shades so each frame costs one fill per shade.
 */
export const makeParticleSphere = (
  { canvas, size, dpr, fps, tint, radius, count: N, floor, scatter, spokes = 0 }: SphereOptions
) => {
  const points = fibonacci(N)
  const ray = rays(N, spokes)
  const ctx = canvas.getContext("2d")
  canvas.width = Math.round(size * dpr)
  canvas.height = Math.round(size * dpr)
  canvas.style.width = `${size}px`
  canvas.style.height = `${size}px`
  const target = { order: 0, heat: 1 }
  const current = { order: 0, heat: 1 }
  const binX = new Float32Array(N)
  const binY = new Float32Array(N)
  const counts = new Uint16Array(LEVELS)
  const levelOf = new Uint8Array(N)
  const sx = new Float32Array(N)
  const sy = new Float32Array(N)
  let beatAt = -Infinity
  let raf = 0
  let last = 0
  let lastFrame = 0
  let enabled = false
  let visible = true
  let idle = false

  const draw = (now: number) => {
    if (!ctx) return
    const dt = lastFrame === 0 ? 16 : Math.min(200, now - lastFrame)
    lastFrame = now
    const k = 1 - Math.exp((-dt * 3) / EASE_MS)
    current.order += (target.order - current.order) * k
    current.heat += (target.heat - current.heat) * k
    const t = now / 1000
    const rot = t * SPIN
    const cr = Math.cos(rot)
    const sr = Math.sin(rot)
    const ct = Math.cos(TILT)
    const st = Math.sin(TILT)
    const half = (size * dpr) / 2
    const R = half * radius
    const spread = 1 - current.order
    counts.fill(0)
    for (let i = 0; i < N; i++) {
      const x = points.unit[i * 3]!
      const y = points.unit[i * 3 + 1]!
      const z = points.unit[i * 3 + 2]!
      const r = R * (1 + spread * (points.jitter[i]! * scatter + 0.06 * Math.sin(t * 0.4 + points.phase[i]!)))
      const x1 = x * cr + z * sr
      const z1 = -x * sr + z * cr
      const y2 = y * ct - z1 * st
      const z2 = y * st + z1 * ct
      const depth = (z2 + 1) / 2
      if (ray.on[i]) {
        // ray dots sit along a screen-space spoke and fade outward; order pulls them back onto the sphere
        const along = R * (0.35 + ray.reach[i]! * scatter * 1.4) * (1 + 0.04 * Math.sin(t * 0.7 + points.phase[i]!))
        const a = ray.angle[i]! + rot * 0.25
        const fade = 1 - ray.reach[i]!
        sx[i] = half + (Math.cos(a) * along) * spread + x1 * R * (1 - spread)
        sy[i] = half + (Math.sin(a) * along) * spread + y2 * R * (1 - spread)
        const level = Math.min(LEVELS - 1, Math.floor((fade * spread + depth * (1 - spread)) * LEVELS))
        levelOf[i] = level
        counts[level]!++
        continue
      }
      const core = spokes > 0 ? R * (1 - spread * 0.3 * points.jitter[i]!) : r
      const level = Math.min(LEVELS - 1, Math.floor(depth * LEVELS))
      levelOf[i] = level
      sx[i] = half + x1 * core
      sy[i] = half + y2 * core
      counts[level]!++
    }
    const starts = new Uint16Array(LEVELS)
    for (let l = 1; l < LEVELS; l++) starts[l] = starts[l - 1]! + counts[l - 1]!
    const cursor = starts.slice()
    for (let i = 0; i < N; i++) {
      const j = cursor[levelOf[i]!]!++
      binX[j] = sx[i]!
      binY[j] = sy[i]!
    }
    const beat = Math.max(0, 1 - (now - beatAt) / BEAT_MS)
    const [r, g, b] = mixHex(theme.fgDim, theme.accent, tint + (1 - tint) * current.heat)
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    for (let l = 0; l < LEVELS; l++) {
      const n = counts[l]!
      if (n === 0) continue
      const d = l / (LEVELS - 1)
      const alpha = Math.min(1, (floor + (1 - floor) * d) * (1 + beat * 0.6))
      const s = (0.6 + 0.6 * d) * dpr * (size > 300 ? 1.6 : 1.2)
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`
      ctx.beginPath()
      for (let j = starts[l]!; j < starts[l]! + n; j++) ctx.rect(binX[j]! - s / 2, binY[j]! - s / 2, s, s)
      ctx.fill()
    }
  }

  const loop = (now: number) => {
    raf = requestAnimationFrame(loop)
    if (now - last < 1000 / (idle ? IDLE_FPS : fps) - 8) return
    last = now
    draw(now)
  }

  const sync = () => {
    cancelAnimationFrame(raf)
    raf = 0
    canvas.hidden = !enabled
    if (!enabled || !visible) return
    if (prefersReducedMotion()) draw(performance.now())
    else raf = requestAnimationFrame(loop)
  }

  new IntersectionObserver(([entry]) => {
    visible = entry?.isIntersecting ?? true
    sync()
  }).observe(canvas)

  const redrawStill = () => {
    if (enabled && prefersReducedMotion()) draw(performance.now())
  }

  return {
    setEnabled: (on: boolean) => {
      enabled = on
      sync()
    },
    setIdle: (on: boolean) => void (idle = on),
    setOrder: (order: number) => {
      target.order = Math.min(1, Math.max(0, order))
      redrawStill()
    },
    setHeat: (heat: number) => {
      target.heat = Math.min(1, Math.max(0, heat))
      redrawStill()
    },
    beat: () => void (beatAt = performance.now())
  }
}

export type ParticleSphere = ReturnType<typeof makeParticleSphere>

/** Mean log10 loss over the recent window, mapped to [0, 1] between 1e-4 and the first loss seen. */
export const lossHeat = (recent: ReadonlyArray<number>, first: number) => {
  const logs = recent.filter((v) => v > 0 && Number.isFinite(v)).map(Math.log10)
  if (logs.length === 0 || !(first > 0)) return 1
  const mean = logs.reduce((a, b) => a + b, 0) / logs.length
  const top = Math.max(Math.log10(first), -3)
  return Math.min(1, Math.max(0, (mean + 4) / (top + 4)))
}
