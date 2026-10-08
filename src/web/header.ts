import type { RunInfo, Status } from "../protocol.ts"

export type Connection = "connecting" | "live" | "down"

const esc = (s: string) => s.replace(/[&<>]/g, (c) => (c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;"))

const kicker = (fields: ReadonlyArray<string>) =>
  fields.filter((f) => f.length > 0).map((f) => `<span>${esc(f)}</span>`).join(`<i class="leader"></i>`)

const backendName = (backend: string) => backend.replace(/^@effect-torch\/(backend-)?/, "")

/** Task fields without a trailing `(backend)` note, since the backend line already names it. */
const taskFields = (task: string, backend: string) =>
  task.split(/,\s*/).map((field) => {
    const note = /^(.*?)\s*\(([^)]*)\)$/.exec(field)
    return note?.[2]?.toLowerCase() === backend.toLowerCase() ? note[1] ?? field : field
  })

const all = <E extends Element>(selector: string) => [...document.querySelectorAll<E>(selector)]

/** Hero header and the compact bar that replaces it once scrolled away. */
export const makeHeader = () => {
  const title = document.getElementById("title")!
  const taskLine = document.getElementById("kicker-task")!
  const backendLine = document.getElementById("kicker-backend")!
  const hero = document.getElementById("hero")!
  const compact = document.getElementById("compact")!
  const statusChips = all<HTMLElement>("[data-chip=status]")
  const stepChips = all<HTMLElement>("[data-chip=step]")
  const connectionChips = all<HTMLElement>("[data-chip=connection]")
  const bars = all<HTMLElement>(".progress")
  const percents = all<HTMLElement>("[data-progress-label]")
  let totalSteps = 0

  new IntersectionObserver(([entry]) => {
    const away = entry ? !entry.isIntersecting : false
    compact.classList.toggle("shown", away)
    compact.setAttribute("aria-hidden", String(!away))
  }).observe(hero)

  const paint = (chips: ReadonlyArray<HTMLElement>, state: string, text: string) => {
    for (const chip of chips) {
      chip.dataset["state"] = state
      chip.textContent = text
    }
  }

  return {
    setRunInfo: (info: RunInfo) => {
      totalSteps = info.totalSteps
      const backend = backendName(info.backend)
      taskLine.innerHTML = kicker(taskFields(info.task, backend))
      backendLine.innerHTML = kicker([
        backend,
        `run ${info.runId}`,
        info.probeEvery > 0 ? `probe every ${info.probeEvery}` : "no probes"
      ])
      title.title = `run ${info.runId}`
      document.title = `effect-ml · ${info.runId}`
    },
    setStep: (step: number) => {
      const text = totalSteps > 0
        ? `step ${step.toLocaleString("en-US")} / ${totalSteps.toLocaleString("en-US")}`
        : `step ${step.toLocaleString("en-US")}`
      for (const chip of stepChips) chip.textContent = text
      const k = totalSteps > 0 ? Math.min(1, step / totalSteps) : 0
      for (const bar of bars) bar.style.setProperty("--k", k.toFixed(4))
      for (const label of percents) {
        label.textContent = totalSteps > 0 ? `${Math.floor(k * 100)}%` : ""
        label.dataset["side"] = k > 0.9 ? "left" : "right"
      }
    },
    setStatus: (status: Status) => {
      paint(statusChips, status.state, status.state === "failed" && status.message ? `failed: ${status.message}` : status.state)
      for (const chip of statusChips) chip.title = status.message ?? ""
    },
    setConnection: (state: Connection) =>
      paint(connectionChips, state, state === "down" ? "retrying…" : state === "live" ? "live" : "connecting")
  }
}
