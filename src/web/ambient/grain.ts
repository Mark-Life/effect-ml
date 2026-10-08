import { prefersReducedMotion } from "../theme.ts"

const TILE = 96
const FRAMES = 8
const SPARSITY = 3.5
const FPS = 14

/** One tile of salt-and-pepper noise; most specks stay near invisible. */
const makeTile = () => {
  const tile = document.createElement("canvas")
  tile.width = TILE
  tile.height = TILE
  const ctx = tile.getContext("2d")
  if (!ctx) return tile
  const image = ctx.createImageData(TILE, TILE)
  const { data } = image
  for (let i = 0; i < data.length; i += 4) {
    const v = Math.random() < 0.5 ? 255 : 0
    data[i] = v
    data[i + 1] = v
    data[i + 2] = v
    data[i + 3] = Math.random() ** SPARSITY * 255
  }
  ctx.putImageData(image, 0, 0)
  return tile
}

/** Film grain: a few noise tiles cycled at film rate with a random offset, drawn at CSS resolution. */
export const makeGrain = (canvas: HTMLCanvasElement) => {
  const ctx = canvas.getContext("2d")
  const patterns = ctx
    ? Array.from({ length: FRAMES }, () => ctx.createPattern(makeTile(), "repeat")).filter((p) => p !== null)
    : []
  let width = 0
  let height = 0
  let raf = 0
  let last = 0
  let frame = 0
  let enabled = false

  const paint = () => {
    if (!ctx || patterns.length === 0 || width === 0) return
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, width, height)
    ctx.translate(-Math.random() * TILE, -Math.random() * TILE)
    ctx.fillStyle = patterns[frame++ % patterns.length]!
    ctx.fillRect(0, 0, width + TILE, height + TILE)
  }

  const loop = (now: number) => {
    raf = requestAnimationFrame(loop)
    if (now - last < 1000 / FPS) return
    last = now
    paint()
  }

  const stop = () => {
    cancelAnimationFrame(raf)
    raf = 0
  }

  new ResizeObserver(([entry]) => {
    if (!entry) return
    width = canvas.width = Math.round(entry.contentRect.width)
    height = canvas.height = Math.round(entry.contentRect.height)
    if (enabled) paint()
  }).observe(canvas)

  return {
    setEnabled: (on: boolean) => {
      enabled = on
      canvas.hidden = !on
      stop()
      if (!on) return
      if (prefersReducedMotion()) paint()
      else raf = requestAnimationFrame(loop)
    }
  }
}
