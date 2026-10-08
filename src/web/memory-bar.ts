import type { MemoryDiagnostics } from "../protocol.ts"

export const formatBytes = (n: number) => {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"]
  let v = n
  let u = 0
  while (Math.abs(v) >= 1024 && u < units.length - 1) {
    v /= 1024
    u++
  }
  return `${u === 0 ? v : v.toFixed(1)} ${units[u]}`
}

const segments = [
  ["external", "externalBytes", 2],
  ["persistent", "persistentBytes", 3],
  ["state", "stateBytes", 4],
  ["output", "outputBytes", 5],
  ["workspace", "workspaceBytes", 3],
  ["transaction", "transactionBytes", 4]
] as const satisfies ReadonlyArray<readonly [string, keyof MemoryDiagnostics, number]>

/** Dot fill at a given pitch; segments are told apart by density, not hue. */
const dots = (pitch: number) =>
  `background-image:radial-gradient(circle,var(--fg-dim) 0.8px,transparent 1.1px);background-size:${pitch}px ${pitch}px`

const HATCH = "background-image:repeating-linear-gradient(45deg,var(--fg-dim) 0 1px,transparent 1px 4px)"

/** One segmented bar of the memory plan, with a peak-live tick and a legend row for every key. */
export const renderMemoryBar = (m: MemoryDiagnostics) => {
  const parts = [
    ...segments.map(([label, key, pitch]) => ({ label, bytes: m[key], style: dots(pitch) })),
    { label: "packingOverhead", bytes: m.packingOverheadBytes, style: HATCH }
  ]
  const total = Math.max(1, parts.reduce((s, p) => s + p.bytes, 0), m.peakLiveBytes)
  const peak = (100 * m.peakLiveBytes) / total
  const bar = parts.filter((p) => p.bytes > 0).map((p) =>
    `<i title="${p.label} ${formatBytes(p.bytes)}" style="flex:${p.bytes} 0 0;${p.style}"></i>`
  ).join("")
  const legend = [...parts, { label: "peakLive", bytes: m.peakLiveBytes, style: "background:var(--accent)" }].map((p) =>
    `<div class="mem-row${p.label === "peakLive" ? " peak" : ""}"><i style="${p.style}"></i><span>${p.label}</span>` +
    `<b>${formatBytes(p.bytes)}</b></div>`
  ).join("")
  return `<div class="mem-bar-wrap"><div class="mem-bar box">${bar}</div>${
    m.peakLiveBytes > 0
      ? `<span class="mem-peak" style="left:${peak}%"><em>peak ${formatBytes(m.peakLiveBytes)}</em></span>`
      : ""
  }</div><div class="mem-legend">${legend}</div>`
}
