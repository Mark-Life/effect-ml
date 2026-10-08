import "@fontsource/anton/400.css"
import "@fontsource/ibm-plex-mono/400.css"
import "@fontsource/ibm-plex-mono/500.css"
import { decodeProbe, decodeServerMessage, type ServerMessage } from "../protocol.ts"
import { makeGrain } from "./ambient/grain.ts"
import { lossHeat, makeParticleSphere } from "./ambient/particle-sphere.ts"
import { bindAmbientToggle } from "./ambient/toggle.ts"
import { makeDiagnosticsPanel } from "./diagnostics-panel.ts"
import { makeGraphView } from "./graph-view.ts"
import { makeHeader } from "./header.ts"
import { makeHeatmaps } from "./heatmap.ts"
import { makeHistograms } from "./histogram.ts"
import { makeLossChart } from "./loss-chart.ts"
import { ambientPref, dotsPref } from "./prefs.ts"
import { prefersReducedMotion } from "./theme.ts"

const $ = <E extends HTMLElement = HTMLElement>(selector: string) => {
  const el = document.querySelector<E>(selector)
  if (!el) throw new Error(`missing element ${selector}`)
  return el
}

const header = makeHeader()
const graph = makeGraphView({ root: $("#graph"), meta: $("#graph-meta") })
const chart = makeLossChart($("#chart"))
const diagnostics = makeDiagnosticsPanel($("#diagnostics"))
const heatmaps = makeHeatmaps($("#heatmaps"))
const histograms = makeHistograms($("#histograms"))

const grain = makeGrain($<HTMLCanvasElement>("#grain"))
const burst = makeParticleSphere({
  canvas: $("#burst"),
  size: 200,
  dpr: Math.min(2, devicePixelRatio || 1),
  fps: 30,
  tint: 1,
  radius: 0.6,
  count: 3000,
  floor: 0.5,
  scatter: 0.6,
  spokes: 11
})
const field = makeParticleSphere({ canvas: $("#field"), size: 560, dpr: 1, fps: 30, tint: 0, radius: 0.5, count: 900, floor: 0.25, scatter: 1.6 })
const spheres = [burst, field]
graph.onWave(() => spheres.forEach((s) => s.beat()))

const applyAmbient = (on: boolean) => {
  grain.setEnabled(on)
  for (const s of spheres) s.setEnabled(on)
  graph.pulses.setEnabled(on && !prefersReducedMotion())
}
applyAmbient(ambientPref.get())
ambientPref.subscribe(applyAmbient)
bindAmbientToggle([...document.querySelectorAll<HTMLElement>("[data-ambient]")])

const dotsToggles = [...document.querySelectorAll<HTMLElement>("[data-dots]")]
const paintDots = (on: boolean) => {
  for (const t of dotsToggles) t.dataset["mode"] = on ? "dots" : "pixels"
}
paintDots(dotsPref.get())
dotsPref.subscribe(paintDots)
for (const t of dotsToggles) t.addEventListener("click", dotsPref.toggle)

let firstLoss = Number.NaN
let recentLoss: Array<number> = []

const handle = (message: ServerMessage) => {
  switch (message._tag) {
    case "RunInfo":
      return header.setRunInfo(message)
    case "Graph":
      return graph.setGraph(message)
    case "Diagnostics":
      return diagnostics.setDiagnostics(message)
    case "Metrics": {
      chart.addMetrics(message.points)
      diagnostics.addMetrics(message.points)
      const last = message.points[message.points.length - 1]
      if (!last) return
      header.setStep(last.step)
      if (Number.isNaN(firstLoss)) firstLoss = message.points[0]!.loss
      recentLoss = recentLoss.concat(message.points.map((p) => p.loss)).slice(-50)
      const heat = lossHeat(recentLoss, firstLoss)
      for (const s of spheres) s.setHeat(heat)
      graph.pulses.onMetrics(last.step)
      return
    }
    case "Eval":
      for (const s of spheres) s.setOrder(message.testAcc)
      return chart.addEval(message)
    case "Status": {
      header.setStatus(message)
      graph.pulses.setTraining(message.state === "training")
      const idle = message.state === "done" || message.state === "failed"
      for (const s of spheres) s.setIdle(idle)
      return
    }
  }
}

const onFrame = (data: unknown) => {
  try {
    if (data instanceof ArrayBuffer) {
      const frame = decodeProbe(data)
      if (frame._tag === "Heatmap") heatmaps.update(frame.header, frame.payload, frame.mask)
      else histograms.update(frame.header, frame.payload)
    } else if (typeof data === "string") {
      handle(decodeServerMessage(JSON.parse(data)))
    }
  } catch (error) {
    console.error("viewer: bad frame", error)
  }
}

const connect = (attempt = 0) => {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`)
  ws.binaryType = "arraybuffer"
  ws.addEventListener("open", () => {
    attempt = 0
    header.setConnection("live")
    // the server replays full state on connect
    chart.reset()
    diagnostics.reset()
    firstLoss = Number.NaN
    recentLoss = []
  })
  ws.addEventListener("message", (e) => onFrame(e.data))
  ws.addEventListener("close", () => {
    header.setConnection("down")
    setTimeout(() => connect(attempt + 1), Math.min(5000, 500 * 2 ** attempt))
  })
}

connect()
