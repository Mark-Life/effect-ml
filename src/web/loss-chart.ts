import uPlot from "uplot"
import "uplot/dist/uPlot.min.css"
import type { EvalPoint, MetricPoint } from "../protocol.ts"
import { evalDelta, plotHeight } from "./chart-format.ts"
import { makeGlowSprite } from "./glow.ts"
import { monoFont, rgba, theme } from "./theme.ts"

const REDRAW_MS = 250
const GLOW_PX = 18
const GROK_ACC = 0.5
/** Step-loss smoothing time constant, in steps. */
const EMA_STEPS = 25
/** Log-scale headroom above the data, in decades; the floor snaps to a whole decade. */
const HEAD_DECADES = 0.15

interface Columns {
  steps: Array<number>
  values: Array<Array<number | null>>
}

const emptyColumns = (series: number): Columns => ({
  steps: [],
  values: Array.from({ length: series }, () => [])
})

/** Inserts or merges a row keyed by step; steps normally arrive in order, so this is usually an append. */
const upsert = (cols: Columns, step: number, row: ReadonlyArray<number | null>) => {
  const last = cols.steps.length - 1
  let i = cols.steps[last] === undefined || step > cols.steps[last]! ? cols.steps.length : -1
  if (i === -1) {
    let lo = 0
    let hi = cols.steps.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (cols.steps[mid]! < step) lo = mid + 1
      else hi = mid
    }
    i = lo
  }
  if (cols.steps[i] === step) {
    row.forEach((v, s) => {
      if (v !== null) cols.values[s]![i] = v
    })
    return
  }
  cols.steps.splice(i, 0, step)
  row.forEach((v, s) => cols.values[s]!.splice(i, 0, v))
}

/** Log-axis labels: plain numbers down to 0.01, exponent form below. */
const logTick = (v: number | null) =>
  v === null ? "" : v >= 0.01 ? String(Number(v.toPrecision(2))) : v.toExponential(0)

const axis = () => ({
  stroke: theme.muted,
  font: monoFont(10),
  grid: { stroke: theme.hair2, dash: [1, 4], width: 1 },
  ticks: { show: false },
  gap: 6
})

interface SeriesSpec {
  readonly label: string
  readonly stroke: string
  readonly width: number
  /** CSS color for the legend swatch. */
  readonly swatch: string
  /** Draws a glowing dot on the latest value. */
  readonly mark?: "glow" | "dot"
  /** Leaves the series out of the legend row. */
  readonly unlisted?: boolean
}

/** Last non-null index of `values` at or before `from`. */
const lastIndex = (values: ArrayLike<number | null | undefined>, from: number) => {
  for (let i = Math.min(from, values.length - 1); i >= 0; i--) if (values[i] !== null && values[i] !== undefined) return i
  return -1
}

const fmtStep = (v: number) => v.toLocaleString("en-US")

const READOUTS = [
  { key: "trainLoss", label: "train loss", kind: "loss", swatch: "var(--fg-dim)" },
  { key: "testLoss", label: "test loss", kind: "loss", swatch: "var(--accent)" },
  { key: "testAcc", label: "test acc", kind: "acc", swatch: "var(--accent)" }
] as const

const fmtReadout = (v: number, kind: "loss" | "acc") =>
  kind === "acc" ? v.toFixed(3) : v !== 0 && Math.abs(v) < 1e-3 ? v.toExponential(2) : v.toPrecision(3)

const makePlot = (
  { target, legend, series, log, grok, format }: {
    target: HTMLElement
    legend: HTMLElement
    series: ReadonlyArray<SeriesSpec>
    log: boolean
    grok: () => number | undefined
    format: (v: number) => string
  }
) => {
  const dpr = uPlot.pxRatio
  const glow = makeGlowSprite(GLOW_PX, dpr)
  const listed = series.flatMap((s, i) => (s.unlisted ? [] : [{ s, i }]))
  legend.innerHTML = `<span class="lg-step">step <b>–</b></span>` + listed.map(({ s }) =>
    `<span><i style="background:${s.swatch}"></i>${s.label} <b>–</b></span>`
  ).join("")
  const values = [...legend.querySelectorAll("b")]

  const updateLegend = (u: uPlot) => {
    const xs = u.data[0]
    const idx = u.cursor.idx ?? xs.length - 1
    if (idx < 0 || xs.length === 0) return
    values[0]!.textContent = fmtStep(xs[idx] ?? 0)
    listed.forEach(({ i: s }, k) => {
      const col = u.data[s + 1]
      if (!col) return
      const i = lastIndex(col, idx)
      values[k + 1]!.textContent = i < 0 ? "–" : format(col[i]!)
    })
  }

  const drawMarks = (u: uPlot) => {
    const { ctx, bbox } = u
    const xs = u.data[0]
    const step = grok()
    ctx.save()
    if (step !== undefined) {
      const x = Math.round(u.valToPos(step, "x", true)) + 0.5
      ctx.strokeStyle = rgba(theme.accent, 0.8)
      ctx.lineWidth = dpr
      ctx.setLineDash([3 * dpr, 3 * dpr])
      ctx.beginPath()
      ctx.moveTo(x, bbox.top)
      ctx.lineTo(x, bbox.top + bbox.height)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = theme.accent
      ctx.font = monoFont(9 * dpr, 500)
      ctx.fillText(`GROK ${fmtStep(step)}`, x + 4 * dpr, bbox.top + 10 * dpr)
    }
    series.forEach((s, i) => {
      if (!s.mark) return
      const col = u.data[i + 1]
      if (!col) return
      const at = lastIndex(col, xs.length - 1)
      if (at < 0) return
      const x = u.valToPos(xs[at]!, "x", true)
      const y = u.valToPos(col[at]!, "y", true)
      if (s.mark === "glow") {
        ctx.globalCompositeOperation = "lighter"
        ctx.drawImage(glow, x - (GLOW_PX * dpr) / 2, y - (GLOW_PX * dpr) / 2)
        ctx.globalCompositeOperation = "source-over"
      }
      ctx.fillStyle = s.mark === "glow" ? theme.hot : s.stroke
      ctx.beginPath()
      ctx.arc(x, y, 2.5 * dpr, 0, Math.PI * 2)
      ctx.fill()
    })
    ctx.restore()
  }

  return new uPlot(
    {
      width: Math.max(300, target.clientWidth),
      height: plotHeight(target.clientWidth),
      legend: { show: false },
      cursor: { sync: { key: "train" }, points: { size: 5 } },
      scales: {
        x: { time: false },
        y: log
          ? {
            distr: 3,
            range: (_u: uPlot, min: number | null, max: number | null): uPlot.Range.MinMax =>
              min === null || max === null || !(min > 0)
                ? [0.01, 10]
                : [10 ** Math.floor(Math.log10(min)), max * 10 ** HEAD_DECADES]
          }
          : { range: [0, 1] }
      },
      axes: [
        { ...axis(), size: 28 },
        { ...axis(), size: 48, ...(log ? { values: (_: uPlot, splits: Array<number>) => splits.map(logTick) } : {}) }
      ],
      series: [
        { label: "step" },
        ...series.map((s) => ({ label: s.label, stroke: s.stroke, width: s.width, spanGaps: true, points: { show: false } }))
      ],
      hooks: { draw: [drawMarks], setCursor: [updateLegend], setData: [updateLegend] }
    },
    [[]],
    target
  )
}

const kicker = (title: string, meta: string) =>
  `<div class="kicker-row"><h3>${title}</h3><i class="leader"></i><span>${meta}</span></div>`

export const makeLossChart = (root: HTMLElement) => {
  root.innerHTML = `<div class="readout">${
    READOUTS.map((r) =>
      `<div class="box" data-readout="${r.key}"><span><i style="background:${r.swatch}"></i>${r.label}</span><b>–</b><small>waiting for eval</small></div>`
    ).join("")
  }</div>
${kicker("loss", "log")}<div class="chart-loss"></div><div class="chart-legend legend-loss"></div>
${kicker("accuracy", "train · test")}<div class="chart-acc"></div><div class="chart-legend legend-acc"></div>`
  const q = (s: string) => root.querySelector<HTMLElement>(s)!
  let grokStep: number | undefined
  const grok = () => grokStep
  const train = { stroke: theme.fgDim, width: 1.25, swatch: "var(--fg-dim)", mark: "dot" } as const
  const test = { stroke: theme.accent, width: 1.5, swatch: "var(--accent)", mark: "glow" } as const
  const lossPlot = makePlot({
    target: q(".chart-loss"),
    legend: q(".legend-loss"),
    log: true,
    grok,
    format: (v) => v.toPrecision(3),
    series: [
      { label: "step", stroke: rgba(theme.fg, 0.1), width: 1, swatch: "var(--faint)", unlisted: true },
      { label: "step loss", stroke: rgba(theme.fg, 0.42), width: 1, swatch: "var(--faint)" },
      { label: "train", ...train },
      { label: "test", ...test }
    ]
  })
  const accPlot = makePlot({
    target: q(".chart-acc"),
    legend: q(".legend-acc"),
    log: false,
    grok,
    format: (v) => v.toFixed(3),
    series: [{ label: "train", ...train }, { label: "test", ...test }]
  })

  let loss = emptyColumns(4)
  let ema: { step: number; log: number } | undefined
  let acc = emptyColumns(2)
  let lastEval: EvalPoint | undefined
  let prevEval: EvalPoint | undefined

  const paintReadout = () => {
    for (const r of READOUTS) {
      const cell = root.querySelector<HTMLElement>(`[data-readout="${r.key}"]`)
      const value = cell?.querySelector("b")
      const delta = cell?.querySelector("small")
      if (!value || !delta) continue
      if (!lastEval) {
        value.textContent = "–"
        delta.textContent = "waiting for eval"
        delta.dataset["trend"] = "flat"
        continue
      }
      const d = evalDelta({ now: lastEval[r.key], before: prevEval?.[r.key], kind: r.kind })
      value.textContent = fmtReadout(lastEval[r.key], r.kind)
      delta.textContent = `${d.text} · step ${fmtStep(lastEval.step)}`
      delta.dataset["trend"] = d.trend
    }
  }
  let dirty = false
  let timer: ReturnType<typeof setTimeout> | undefined

  const redraw = () => {
    timer = undefined
    if (!dirty) return
    dirty = false
    lossPlot.setData([loss.steps, ...loss.values])
    accPlot.setData([acc.steps, ...acc.values])
  }
  const schedule = () => {
    dirty = true
    timer ??= setTimeout(redraw, REDRAW_MS)
  }

  new ResizeObserver(() => {
    const width = Math.max(300, root.clientWidth)
    const height = plotHeight(width)
    lossPlot.setSize({ width, height })
    accPlot.setSize({ width, height })
  }).observe(root)
  void document.fonts.ready.then(() => {
    lossPlot.redraw(false, true)
    accPlot.redraw(false, true)
  })

  const positive = (v: number) => (v > 0 && Number.isFinite(v) ? v : null)

  return {
    addMetrics: (points: ReadonlyArray<MetricPoint>) => {
      for (const p of points) {
        const v = positive(p.loss)
        if (v !== null && (!ema || p.step > ema.step)) {
          const k = ema ? 1 - Math.exp(-(p.step - ema.step) / EMA_STEPS) : 1
          const log = ema ? ema.log + (Math.log(v) - ema.log) * k : Math.log(v)
          ema = { step: p.step, log }
        }
        upsert(loss, p.step, [v, ema && ema.step === p.step ? Math.exp(ema.log) : null, null, null])
      }
      schedule()
    },
    addEval: (e: EvalPoint) => {
      upsert(loss, e.step, [null, null, positive(e.trainLoss), positive(e.testLoss)])
      upsert(acc, e.step, [e.trainAcc, e.testAcc])
      if (e.testAcc >= GROK_ACC && (grokStep === undefined || e.step < grokStep)) grokStep = e.step
      if (!lastEval || e.step > lastEval.step) {
        prevEval = lastEval
        lastEval = e
        paintReadout()
      }
      schedule()
    },
    reset: () => {
      loss = emptyColumns(4)
      ema = undefined
      acc = emptyColumns(2)
      grokStep = undefined
      lastEval = undefined
      prevEval = undefined
      paintReadout()
      schedule()
    }
  }
}
