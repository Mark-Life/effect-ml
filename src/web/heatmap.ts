import type { HeatmapHeader } from "../protocol.ts"
import { buildLut, buildLutAtlas, buildTwoTone, LUT_SIZE } from "./colormap-lut.ts"
import { colormapWgsl, PARAMS_BYTES } from "./colormap.wgsl.ts"
import { isTwoLevel, logit, RAMPS, robustRange, sigmoid, twoLevelSplit } from "./heatmap-range.ts"
import { dotsPref } from "./prefs.ts"
import { divergingStops, emberStops, twoToneStops } from "./theme.ts"

const TILE_W = 226
const DOT_CELL = 4

interface Gpu {
  readonly device: GPUDevice
  readonly format: GPUTextureFormat
  readonly pipeline: GPURenderPipeline
  readonly layout: GPUBindGroupLayout
  readonly lut: GPUTexture
}

interface Frame {
  readonly header: HeatmapHeader
  readonly payload: Float32Array
  readonly mask: Float32Array | undefined
}

interface Tile {
  readonly title: HTMLElement
  readonly meta: HTMLElement
  readonly canvas: HTMLCanvasElement
  readonly scale: HTMLElement
  readonly bar: HTMLElement
  readonly lo: HTMLElement
  readonly hi: HTMLElement
  readonly levels: HTMLElement
  readonly crossH: HTMLElement
  readonly crossV: HTMLElement
  readonly context: GPUCanvasContext
  readonly uniforms: GPUBuffer
  texture?: GPUTexture
  bindGroup?: GPUBindGroup
  last?: Frame
  /** Interleaved (value, mask) texels for the rg32float texture. */
  texels?: Float32Array
  baseMeta: string
  hover?: { row: number; col: number } | undefined
}

const MODES = { sequential: 0, diverging: 1, probability: 0, high: 2, low: 3 } as const
const lutRows = [buildLut(emberStops), buildLut(divergingStops), buildTwoTone(twoToneStops.high), buildTwoTone(twoToneStops.low)]
const lutAtlas = buildLutAtlas(lutRows)

/** Data URLs of the LUT rows, for the colorbars. */
const colorbars = (() => {
  const canvas = document.createElement("canvas")
  canvas.width = LUT_SIZE
  canvas.height = 1
  const ctx = canvas.getContext("2d")
  return lutRows.map((_, row) => {
    if (!ctx) return ""
    ctx.putImageData(new ImageData(new Uint8ClampedArray(lutAtlas.slice(row * LUT_SIZE * 4, (row + 1) * LUT_SIZE * 4)), LUT_SIZE, 1), 0, 0)
    return canvas.toDataURL()
  })
})()

const initGpu = async (): Promise<Gpu | string> => {
  if (!("gpu" in navigator)) return "WebGPU unavailable: navigator.gpu is missing"
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) return "WebGPU unavailable: no adapter"
  const device = await adapter.requestDevice()
  const format = navigator.gpu.getPreferredCanvasFormat()
  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } }
    ]
  })
  const module = device.createShaderModule({ code: colormapWgsl })
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" }
  })
  const lut = device.createTexture({
    size: [LUT_SIZE, lutRows.length],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  })
  device.queue.writeTexture({ texture: lut }, lutAtlas, { bytesPerRow: LUT_SIZE * 4 }, [LUT_SIZE, lutRows.length])
  return { device, format, pipeline, layout, lut }
}

export const fmt = (v: number) =>
  Math.abs(v) >= 1e3 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(1) : v.toFixed(3)

/** Probabilities close to 1 read as `1−ε`, so 0.9999 and 0.99 do not both print as 1.000. */
const fmtProbability = (p: number) => (p > 0.99 && p < 1 ? `1−${fmt(1 - p)}` : fmt(p))

/** Four significant digits for hover readouts. */
export const fmtValue = (v: number) =>
  !Number.isFinite(v)
    ? String(v)
    : Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3)
    ? v.toExponential(2)
    : String(Number(v.toPrecision(4)))

const pct = (share: number) => `${share > 0 && share < 0.01 ? "<1" : Math.round(share * 100)}%`

/** Value under the pointer, phrased with the map's axis names and level names when it has them. */
const readout = ({ header, payload, mask }: Frame, { row, col }: { row: number; col: number }) => {
  const i = row * header.cols + col
  const v = payload[i] ?? Number.NaN
  const at = header.axes ? `${header.axes.rows}=${row} ${header.axes.cols}=${col}` : `row ${row} col ${col}`
  const mid = header.highlight !== undefined
    ? (Math.min(header.min, 0) + Math.max(header.max, 1)) / 2
    : (header.min + header.max) / 2
  const value = header.levels && Number.isFinite(v) ? (v >= mid ? header.levels.high : header.levels.low) : fmtValue(v)
  const group = header.mask && mask ? `${mask[i]! >= 0.5 ? header.mask.dimmed : header.mask.shown} · ` : ""
  return `${at} → ${group}${value}`
}

/** Whole-pixel scale when at least 2× fits, so cells stay crisp; otherwise stretch to the tile width. */
const displaySize = (rows: number, cols: number) => {
  const k = Math.floor(TILE_W / cols)
  if (k >= 2) return { width: cols * k, height: Math.min(Math.round(TILE_W * 1.5), rows * k) }
  return { width: TILE_W, height: Math.round(Math.min(TILE_W * 1.5, Math.max(14, (TILE_W * rows) / cols))) }
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className: string) =>
  Object.assign(document.createElement(tag), { className })

export const makeHeatmaps = (root: HTMLElement) => {
  const status = el("div", "panel-message")
  status.textContent = "initializing WebGPU…"
  const grid = el("div", "tiles")
  root.replaceChildren(status, grid)

  const tiles = new Map<string, Tile>()
  const pending = new Map<string, Frame>()
  let gpu: Gpu | undefined

  const makeTile = (g: Gpu, name: string): Tile => {
    const figure = el("figure", "tile")
    const title = el("figcaption", "")
    title.textContent = name
    const frame = el("div", "tile-frame")
    const canvas = document.createElement("canvas")
    frame.append(canvas)
    const crossH = el("i", "crosshair h")
    const crossV = el("i", "crosshair v")
    frame.append(crossH, crossV)
    const scale = el("div", "colorbar")
    const lo = el("span", "")
    const bar = el("i", "")
    const hi = el("span", "")
    scale.append(lo, bar, hi)
    const levels = el("div", "levels")
    const meta = el("div", "tile-meta")
    figure.append(title, frame, scale, levels, meta)
    grid.append(figure)
    const context = canvas.getContext("webgpu")
    if (!context) throw new Error(`webgpu context unavailable for ${name}`)
    context.configure({ device: g.device, format: g.format, alphaMode: "opaque" })
    const uniforms = g.device.createBuffer({
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })
    const tile: Tile = { title, meta, canvas, scale, bar, lo, hi, levels, crossH, crossV, context, uniforms, baseMeta: "" }
    canvas.addEventListener("pointermove", (e) => {
      if (!tile.last) return
      const r = canvas.getBoundingClientRect()
      const { rows, cols } = tile.last.header
      const col = Math.min(cols - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * cols)))
      const row = Math.min(rows - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * rows)))
      tile.hover = { row, col }
      paintHover(tile)
    })
    canvas.addEventListener("pointerleave", () => {
      tile.hover = undefined
      paintHover(tile)
    })
    return tile
  }

  const paintHover = (tile: Tile) => {
    const { hover, last } = tile
    tile.crossH.hidden = tile.crossV.hidden = !hover || !last
    tile.meta.classList.toggle("is-readout", !!hover && !!last)
    if (!hover || !last) {
      tile.meta.textContent = tile.baseMeta
      return
    }
    const { rows, cols } = last.header
    const w = tile.canvas.clientWidth
    const h = tile.canvas.clientHeight
    tile.crossV.style.left = `${1 + Math.floor(((hover.col + 0.5) / cols) * w)}px`
    tile.crossH.style.top = `${1 + Math.floor(((hover.row + 0.5) / rows) * h)}px`
    tile.meta.textContent = readout(last, hover)
  }

  const draw = (g: Gpu, frame: Frame) => {
    const { header, payload, mask } = frame
    const tile = tiles.get(header.name) ?? makeTile(g, header.name)
    tiles.set(header.name, tile)
    tile.last = frame
    const { rows, cols } = header
    if (!tile.texture || tile.texture.width !== cols || tile.texture.height !== rows) {
      tile.texture?.destroy()
      tile.texture = g.device.createTexture({
        size: [cols, rows],
        format: "rg32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      })
      tile.bindGroup = g.device.createBindGroup({
        layout: g.layout,
        entries: [
          { binding: 0, resource: tile.texture.createView() },
          { binding: 1, resource: { buffer: tile.uniforms } },
          { binding: 2, resource: g.lut.createView() }
        ]
      })
      const size = displaySize(rows, cols)
      const dpr = window.devicePixelRatio || 1
      tile.canvas.style.width = `${size.width}px`
      tile.canvas.style.height = `${size.height}px`
      tile.canvas.width = Math.round(size.width * dpr)
      tile.canvas.height = Math.round(size.height * dpr)
    }
    const texels = tile.texels?.length === rows * cols * 2 ? tile.texels : new Float32Array(rows * cols * 2)
    tile.texels = texels
    for (let i = 0; i < rows * cols; i++) {
      texels[i * 2] = payload[i]!
      texels[i * 2 + 1] = mask?.[i] ?? 0
    }
    g.device.queue.writeTexture({ texture: tile.texture }, texels, { bytesPerRow: cols * 8 }, [cols, rows])

    // a `highlight` map is boolean even when every cell agrees, so all-correct stays dark instead of a white block
    const twoLevel = header.highlight !== undefined || isTwoLevel(payload, header)
    const { lo, hi, constant, ramp } = header.highlight !== undefined
      ? {
        lo: Math.min(header.min, 0),
        hi: Math.max(header.max, 1),
        constant: header.min === header.max,
        ramp: "linear" as const
      }
      : twoLevel
      ? { lo: header.min, hi: header.max, constant: false, ramp: "linear" as const }
      : robustRange(payload, header)
    const split = twoLevel
      ? twoLevelSplit({ payload, lo, hi, prefer: header.highlight ?? "high", mask: header.mask ? mask : undefined })
      : undefined
    const diverging = !twoLevel && header.scale === "diverging"
    const mode = split ? MODES[split.lit] : MODES[header.scale]
    const params = new ArrayBuffer(PARAMS_BYTES)
    const f32 = new Float32Array(params)
    const u32 = new Uint32Array(params)
    f32[0] = lo
    f32[1] = hi
    u32[2] = mode
    u32[3] = dotsPref.get() ? 1 : 0
    f32[4] = DOT_CELL * (window.devicePixelRatio || 1)
    f32[5] = tile.canvas.width
    f32[6] = tile.canvas.height
    u32[7] = RAMPS[ramp]
    g.device.queue.writeBuffer(tile.uniforms, 0, params)

    const encoder = g.device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: tile.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }]
    })
    pass.setPipeline(g.pipeline)
    if (tile.bindGroup) pass.setBindGroup(0, tile.bindGroup)
    pass.draw(3)
    pass.end()
    g.device.queue.submit([encoder.finish()])

    const axes = header.axes ? ` · rows ${header.axes.rows}, cols ${header.axes.cols}` : ""
    tile.title.textContent = header.name
    tile.baseMeta = `step ${header.step} · [${header.sourceShape.join(",")}]→${rows}×${cols}${axes}${
      constant && !split ? " · constant" : ""
    }${ramp === "linear" ? "" : ` · ${ramp}`}`
    tile.scale.hidden = !!split
    tile.levels.hidden = !split
    if (split) {
      const names = header.levels ?? { low: fmt(lo), high: fmt(hi) }
      const group = header.mask ? `${header.mask.shown} ` : ""
      const swatch = (level: "low" | "high", share: number) => {
        const span = el("span", split.lit === level ? "lit" : "")
        span.append(el("i", ""), `${group}${names[level]} ${pct(share)}`)
        return span
      }
      const dimmed = header.mask ? [Object.assign(el("span", "dim"), { textContent: header.mask.dimmed })] : []
      if (dimmed[0]) dimmed[0].prepend(el("i", ""))
      tile.levels.replaceChildren(...dimmed, swatch("low", split.lowShare), swatch("high", split.highShare))
    } else {
      tile.bar.style.backgroundImage = `url(${colorbars[mode]})`
      // `≤` and `≥` mark ends clipped at a percentile
      const logits = ramp === "logit"
      const [min, max] = logits ? [logit(header.min), logit(header.max)] : [header.min, header.max]
      const label = (v: number) => (logits ? fmtProbability(sigmoid(v)) : fmt(v))
      tile.lo.textContent = `${lo > min + 1e-9 ? "≤" : ""}${label(lo)}`
      tile.hi.textContent = `${hi < max - 1e-9 ? "≥" : ""}${diverging ? "+" : ""}${label(hi)}`
    }
    paintHover(tile)
  }

  const flush = () => {
    if (!gpu) return
    for (const frame of pending.values()) draw(gpu, frame)
    pending.clear()
  }

  let message: HTMLDivElement | undefined
  const showMessage = (text: string) => {
    message ??= el("div", "panel-message")
    if (!message.isConnected) root.prepend(message)
    message.textContent = text
  }

  initGpu().then(
    (result) => {
      if (typeof result === "string") {
        status.textContent = result
        return
      }
      gpu = result
      status.remove()
      result.device.addEventListener("uncapturederror", (e) => {
        console.error("WebGPU error:", e.error.message)
        showMessage(e.error.message)
      })
      void result.device.lost.then((info) => {
        gpu = undefined
        showMessage(`GPU device lost: ${info.message}. Reload the page.`)
      })
      flush()
    },
    (error: unknown) => {
      status.textContent = `WebGPU init failed: ${String(error)}`
    }
  )

  let scheduled = false
  const schedule = () => {
    if (scheduled) return
    scheduled = true
    requestAnimationFrame(() => {
      scheduled = false
      flush()
    })
  }

  dotsPref.subscribe(() => {
    for (const tile of tiles.values()) if (tile.last) pending.set(tile.last.header.name, tile.last)
    schedule()
  })

  return {
    update: (header: HeatmapHeader, payload: Float32Array, mask?: Float32Array) => {
      pending.set(header.name, { header, payload, mask })
      schedule()
    }
  }
}
