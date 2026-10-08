import type { HistogramHeader } from "../protocol.ts"
import { fmt, fmtValue } from "./heatmap.ts"
import { dotsPref } from "./prefs.ts"
import { rgba, theme } from "./theme.ts"

const W = 226
const H = 70
const TOP = 4
const PITCH = 3
const DOT = 1.5

interface Tile {
  readonly canvas: HTMLCanvasElement
  readonly meta: HTMLElement
  readonly nan: HTMLElement
  last?: { header: HistogramHeader; counts: Uint32Array }
  hover?: number | undefined
}

const drawBars = (
  canvas: HTMLCanvasElement,
  header: HistogramHeader,
  counts: Uint32Array,
  { dots, hover }: { dots: boolean; hover: number | undefined }
) => {
  const dpr = window.devicePixelRatio || 1
  canvas.width = W * dpr
  canvas.height = H * dpr
  canvas.style.width = `${W}px`
  canvas.style.height = `${H}px`
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, W, H)
  const peak = Math.max(1, ...counts)
  const barW = W / counts.length
  // sqrt scale keeps tails visible next to a tall peak
  const heightOf = (c: number) => Math.sqrt(c / peak) * (H - TOP - 1)
  ctx.fillStyle = rgba(theme.fgDim, dots ? 0.8 : 0.55)
  ctx.beginPath()
  counts.forEach((c, i) => {
    const h = heightOf(c)
    if (!dots) {
      if (h > 0) ctx.rect(i * barW + 0.5, H - 1 - h, Math.max(1, barW - 1), h)
      return
    }
    const x = i * barW + barW / 2 - DOT / 2
    const n = Math.round(h / PITCH)
    for (let k = 0; k < n; k++) ctx.rect(x, H - 1 - PITCH * (k + 1), DOT, DOT)
  })
  ctx.fill()
  if (hover !== undefined) {
    ctx.fillStyle = rgba(theme.fg, 0.06)
    ctx.fillRect(hover * barW, 0, barW, H - 1)
    const h = heightOf(counts[hover] ?? 0)
    ctx.fillStyle = theme.fg
    if (h > 0) ctx.fillRect(hover * barW + 0.5, H - 1 - h, Math.max(1, barW - 1), h)
  }
  ctx.fillStyle = theme.hair
  ctx.fillRect(0, H - 1, W, 1)
  if (header.min < 0 && header.max > 0) {
    const x = Math.round(((0 - header.min) / (header.max - header.min)) * W) + 0.5
    ctx.fillStyle = rgba(theme.accent, 0.6)
    ctx.fillRect(x - 0.5, 0, 1, H - 1)
  }
}

export const makeHistograms = (root: HTMLElement) => {
  const grid = Object.assign(document.createElement("div"), { className: "tiles" })
  root.replaceChildren(grid)
  const tiles = new Map<string, Tile>()

  const makeTile = (name: string): Tile => {
    const figure = Object.assign(document.createElement("figure"), { className: "tile" })
    const title = document.createElement("figcaption")
    title.textContent = name
    const frame = Object.assign(document.createElement("div"), { className: "tile-frame" })
    const canvas = document.createElement("canvas")
    const nan = Object.assign(document.createElement("span"), { className: "nan-tag", hidden: true })
    frame.append(canvas, nan)
    const meta = Object.assign(document.createElement("div"), { className: "tile-meta" })
    figure.append(title, frame, meta)
    grid.append(figure)
    const tile: Tile = { canvas, meta, nan }
    canvas.addEventListener("pointermove", (e) => {
      if (!tile.last) return
      const r = canvas.getBoundingClientRect()
      const bins = tile.last.counts.length
      const bin = Math.min(bins - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * bins)))
      if (bin === tile.hover) return
      tile.hover = bin
      paint(tile, tile.last.header, tile.last.counts)
    })
    canvas.addEventListener("pointerleave", () => {
      tile.hover = undefined
      if (tile.last) paint(tile, tile.last.header, tile.last.counts)
    })
    return tile
  }

  const binText = (header: HistogramHeader, counts: Uint32Array, bin: number) => {
    const width = (header.max - header.min) / counts.length
    const lo = header.min + bin * width
    // enough decimals to tell neighbouring bin edges apart
    const digits = width > 0 ? Math.min(8, Math.max(0, Math.ceil(-Math.log10(width)) + 1)) : 3
    const edge = (v: number) => (Math.abs(v) >= 1e4 ? fmtValue(v) : v.toFixed(digits))
    return `[${edge(lo)}, ${edge(lo + width)}) → ${(counts[bin] ?? 0).toLocaleString("en-US")}`
  }

  const paint = (tile: Tile, header: HistogramHeader, counts: Uint32Array) => {
    tile.last = { header, counts }
    const hover = tile.hover !== undefined && tile.hover < counts.length ? tile.hover : undefined
    drawBars(tile.canvas, header, counts, { dots: dotsPref.get(), hover })
    const bad = header.nonFinite > 0
    tile.nan.hidden = !bad
    tile.nan.textContent = `NaN×${header.nonFinite}`
    tile.meta.textContent = hover !== undefined
      ? binText(header, counts, hover)
      : `step ${header.step} · [${header.sourceShape.join(",")}] · ${fmt(header.min)}…${fmt(header.max)}${
        bad ? ` · ${header.nonFinite} non-finite` : ""
      }`
    tile.meta.classList.toggle("warn", bad && hover === undefined)
    tile.meta.classList.toggle("is-readout", hover !== undefined)
  }

  dotsPref.subscribe(() => {
    for (const tile of tiles.values()) if (tile.last) paint(tile, tile.last.header, tile.last.counts)
  })

  return {
    update: (header: HistogramHeader, payload: Uint32Array) => {
      const tile = tiles.get(header.name) ?? makeTile(header.name)
      tiles.set(header.name, tile)
      paint(tile, header, payload)
    }
  }
}
