import type { DiagnosticsMessage, MetricPoint, ProgramId } from "../protocol.ts"
import { makeGlowSprite } from "./glow.ts"
import { formatBytes, renderMemoryBar } from "./memory-bar.ts"
import { rgba, theme } from "./theme.ts"

const SPARK_POINTS = 600
const SPARK_H = 40
/** Share of the sparkline, at the live end, that gets a faint fill. */
const SPARK_TAIL = 0.08

const escape = (s: string) => s.replace(/[&<>"]/g, (c) => (c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&quot;"))

const subhead = (title: string, meta = "") =>
  `<div class="kicker-row"><h3>${title}</h3><i class="leader"></i><span>${meta}</span></div>`

const halftoneBars = (rows: ReadonlyArray<{ label: string; value: number }>) => {
  const peak = Math.max(1, ...rows.map((r) => r.value))
  return rows.map((r, i) =>
    `<div class="bar-row${i === 0 ? " top" : ""}"><span class="bar-label" title="${escape(r.label)}">${
      escape(r.label)
    }</span><span class="bar" style="width:${(100 * r.value) / peak}%"></span><span class="bar-value">${r.value}</span></div>`
  ).join("")
}

const renderDiagnostics = ({ diagnostics: d, signature }: DiagnosticsMessage) => {
  const instructions = [...d.instructions].sort((a, b) => b.count - a.count)
  const total = d.instructions.reduce((s, i) => s + i.count, 0)
  const counters = [
    ["semantic nodes", `${d.semanticNodesBeforeOptimization}<small>→</small>${d.semanticNodesAfterOptimization}`],
    ["instructions", String(total)],
    ["pipelines", String(d.pipelineCount)],
    ["commands", String(d.commandCount)],
    ["syncs", String(d.synchronizationCount)]
  ]
  const phases = d.compilePhases ?? []
  const phaseTotal = phases.reduce((s, p) => s + p.nanoseconds, 0) / 1e6
  return `
<div class="counters">${counters.map(([k, v]) => `<div class="box"><b>${v}</b><span>${k}</span></div>`).join("")}</div>
${subhead("instructions by kind", `${instructions.length} kinds`)}
<div class="bars">${halftoneBars(instructions.map((i) => ({ label: i.kind, value: i.count })))}</div>
${subhead("memory plan")}
${renderMemoryBar(d.memory)}
${
    phases.length > 0
      ? `${subhead("compile phases", `${phaseTotal.toFixed(2)} ms`)}<div class="phases">${
        phases.map((p) => {
          const ms = p.nanoseconds / 1e6
          return `<div class="phase${ms > 1 ? " slow" : ""}"><span>${escape(p.phase)}</span><i class="leader"></i><b>${
            ms.toFixed(2)
          } ms</b></div>`
        }).join("")
      }</div>`
      : ""
  }
<button class="signature" title="${escape(signature)}" data-signature="${escape(signature)}">${escape(signature)}</button>`
}

const drawSparkline = (canvas: HTMLCanvasElement, values: ReadonlyArray<number>, glow: HTMLCanvasElement) => {
  const dpr = window.devicePixelRatio || 1
  const width = canvas.clientWidth || 300
  canvas.width = width * dpr
  canvas.height = SPARK_H * dpr
  const ctx = canvas.getContext("2d")
  if (!ctx || values.length < 2) return
  ctx.scale(dpr, dpr)
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const span = hi - lo || 1
  const pad = 6
  const at = (v: number, i: number) => ({
    x: pad + (i / (values.length - 1)) * (width - pad * 2),
    y: SPARK_H - 4 - ((v - lo) / span) * (SPARK_H - 10)
  })
  ctx.beginPath()
  values.forEach((v, i) => {
    const p = at(v, i)
    if (i === 0) ctx.moveTo(p.x, p.y)
    else ctx.lineTo(p.x, p.y)
  })
  ctx.strokeStyle = rgba(theme.accent, 0.85)
  ctx.lineWidth = 1
  ctx.stroke()
  const from = Math.floor((values.length - 1) * (1 - SPARK_TAIL))
  const start = at(values[from]!, from)
  ctx.beginPath()
  ctx.moveTo(start.x, SPARK_H)
  for (let i = from; i < values.length; i++) {
    const p = at(values[i]!, i)
    ctx.lineTo(p.x, p.y)
  }
  ctx.lineTo(width - pad, SPARK_H)
  ctx.closePath()
  const fade = ctx.createLinearGradient(start.x, 0, width - pad, 0)
  fade.addColorStop(0, rgba(theme.accent, 0))
  fade.addColorStop(1, rgba(theme.accent, 0.14))
  ctx.fillStyle = fade
  ctx.fill()
  const last = at(values[values.length - 1]!, values.length - 1)
  ctx.globalCompositeOperation = "lighter"
  ctx.drawImage(glow, last.x - 9, last.y - 9, 18, 18)
  ctx.globalCompositeOperation = "source-over"
  ctx.fillStyle = theme.hot
  ctx.beginPath()
  ctx.arc(last.x, last.y, 2, 0, Math.PI * 2)
  ctx.fill()
}

export const makeDiagnosticsPanel = (root: HTMLElement) => {
  root.innerHTML = `
<div class="diag-toolbar">
  <label class="select tag"><select data-program><option value="train-step">train-step</option><option value="probe">probe</option></select></label>
  <span class="live-memory"></span>
</div>
<canvas class="sparkline"></canvas>
<div class="diag-body"><div class="panel-message">waiting for diagnostics…</div></div>`
  const select = root.querySelector<HTMLSelectElement>("[data-program]")!
  const body = root.querySelector<HTMLElement>(".diag-body")!
  const live = root.querySelector<HTMLElement>(".live-memory")!
  const spark = root.querySelector<HTMLCanvasElement>(".sparkline")!
  const glow = makeGlowSprite(18, window.devicePixelRatio || 1)
  const messages = new Map<ProgramId, DiagnosticsMessage>()
  let memory: Array<number> = []
  let stepMs: Array<number> = []
  let sparkQueued = false

  const render = () => {
    const programId: ProgramId = select.value === "probe" ? "probe" : "train-step"
    const message = messages.get(programId)
    body.innerHTML = message
      ? renderDiagnostics(message)
      : `<div class="panel-message">no diagnostics for ${programId}</div>`
  }
  select.addEventListener("change", render)
  body.addEventListener("click", (e) => {
    const button = e.target instanceof Element ? e.target.closest<HTMLElement>("[data-signature]") : null
    if (!button) return
    void navigator.clipboard.writeText(button.dataset["signature"] ?? "").then(() => {
      button.classList.add("copied")
      setTimeout(() => button.classList.remove("copied"), 1000)
    })
  })

  return {
    setDiagnostics: (message: DiagnosticsMessage) => {
      messages.set(message.programId, message)
      if (messages.size === 1) select.value = message.programId
      render()
    },
    addMetrics: (points: ReadonlyArray<MetricPoint>) => {
      if (points.length === 0) return
      memory = memory.concat(points.map((p) => p.memoryBytes)).slice(-SPARK_POINTS)
      stepMs = stepMs.concat(points.map((p) => p.stepMs)).slice(-50)
      const last = points[points.length - 1]!
      const avg = stepMs.reduce((a, b) => a + b, 0) / stepMs.length
      live.innerHTML = `memory <b>${formatBytes(last.memoryBytes)}</b> · <b>${avg.toFixed(1)}</b> ms/step`
      if (sparkQueued) return
      sparkQueued = true
      requestAnimationFrame(() => {
        sparkQueued = false
        drawSparkline(spark, memory, glow)
      })
    },
    reset: () => {
      memory = []
      stepMs = []
    }
  }
}
