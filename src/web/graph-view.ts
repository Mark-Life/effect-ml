import dagre, { type EdgeLabel, type GraphLabel, type NodeLabel } from "@dagrejs/dagre"
import type { GraphId, GraphMessage } from "../protocol.ts"
import { collapse, type ViewGraph, type ViewNode } from "./graph-model.ts"
import { attachPanZoom, fitView, MIN_FIT, MIN_LEGIBLE, screenTransform, type ViewBox, zoomAt } from "./graph-panzoom.ts"
import { activePath, edgeWindows, nodeDepths, type Point, samplePolyline, smoothPath } from "./graph-path.ts"
import { makePulses, type PulseEdge } from "./graph-pulses.ts"
import { families, glyphSymbols } from "./op-family.ts"
import { renderTooltip } from "./graph-tooltip.ts"

const NODE_BUDGET = 1500
const CHAR_W = 6.3
const DETAIL_CHAR_W = 5.7
const MIN_NODE_W = 96
const SVG_NS = "http://www.w3.org/2000/svg"
/** Full boxes; then screen-sized chips with the op label cut to the room beside each node; then dots. */
const LOD = [
  { min: MIN_LEGIBLE, cls: "lod-near" },
  { min: MIN_FIT, cls: "lod-fit" },
  { min: 0, cls: "lod-far" }
] as const
/** Screen px per character of the 10 px mono chip label, the chip's side padding, height and widest label. */
const FIT_CHAR_PX = 6
const CHIP_PAD_PX = 5
const CHIP_H_PX = 15
const CHIP_MAX_PX = 150
/** Screen gap kept between chips of neighbouring nodes in one rank. */
const CHIP_GAP_PX = 6
const EDGE_ALPHA = { short: 0.3, long: 0.1 } as const
const ZOOM_STEP = 1.3

export const escape = (s: string) =>
  s.replace(/[&<>"]/g, (c) => (c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&quot;"))

const chipText = (n: ViewNode) => n.chips.map((c) => `⟨${c}⟩`).join(" ")

/** Width follows the op label only; detail and chips are clipped to it (the tooltip has them in full). */
const nodeSize = (n: ViewNode) => ({
  width: Math.ceil(Math.max(MIN_NODE_W, n.label.length * CHAR_W + 36)),
  height: n.chips.length > 0 ? 54 : 40
})

const clip = (text: string, width: number) => {
  const max = Math.floor((width - 20) / DETAIL_CHAR_W)
  return text.length <= max ? text : `${text.slice(0, Math.max(1, max - 1))}…`
}

interface Placed {
  readonly node: ViewNode
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  /** Distance to the nearest node of the same rank, in graph units; the chip label may use that much. */
  readonly room: number
}

/** Nodes of a rank share a center `y`; each gets the distance to its nearest rank neighbour. */
const withRoom = (placed: ReadonlyArray<Omit<Placed, "room">>): Array<Placed> => {
  const ranks = new Map<number, Array<number>>()
  for (const p of placed) {
    const key = Math.round(p.y)
    ranks.set(key, [...(ranks.get(key) ?? []), p.x])
  }
  return placed.map((p) => {
    const xs = ranks.get(Math.round(p.y)) ?? []
    const room = xs.reduce((min, x) => (x === p.x ? min : Math.min(min, Math.abs(x - p.x))), Infinity)
    return { ...p, room }
  })
}

const layout = (graph: ViewGraph) => {
  const g = new dagre.graphlib.Graph<GraphLabel, NodeLabel, EdgeLabel>()
  g.setGraph({ rankdir: "TB", nodesep: 14, ranksep: 30, marginx: 32, marginy: 32 })
  g.setDefaultEdgeLabel(() => ({}))
  for (const n of graph.nodes) g.setNode(String(n.id), nodeSize(n))
  for (const e of graph.edges) g.setEdge(String(e.from), String(e.to))
  dagre.layout(g)
  const placed = withRoom(graph.nodes.map((node) => {
    const { x = 0, y = 0, width, height } = g.node(String(node.id))
    return { node, x, y, width, height }
  }))
  const byId = new Map(placed.map((p) => [p.node.id, p]))
  const edges = graph.edges.map((e) => {
    const from = byId.get(e.from)
    const to = byId.get(e.to)
    const raw: ReadonlyArray<Point> = g.edge(String(e.from), String(e.to))?.points ?? []
    const points = from && to && raw.length >= 2
      ? [{ x: from.x, y: from.y + from.height / 2 }, ...raw.slice(1, -1), { x: to.x, y: to.y - to.height / 2 }]
      : raw
    return { ...e, points }
  })
  const label = g.graph()
  return { placed, edges, width: Number(label.width ?? 100), height: Number(label.height ?? 100) }
}

type Laid = ReturnType<typeof layout>

const renderNode = ({ node, x, y, width, height }: Placed, ports: { in: boolean; out: boolean }) => {
  const left = x - width / 2
  const top = y - height / 2
  const taps = node.taps.join(", ")
  const tapTag = taps
    ? `<g class="tap"><rect x="${left}" y="${top - 16}" width="${taps.length * 5.6 + 12}" height="13" rx="1"/>` +
      `<text x="${left + 6}" y="${top - 6.5}">${escape(taps.toUpperCase())}</text></g>`
    : ""
  const chips = node.chips.length > 0
    ? `<text class="chips" x="${left + 10}" y="${top + 44}">${escape(clip(chipText(node), width))}</text>`
    : ""
  const tick = taps ? `<line class="tap-tick" x1="${left + 6}" y1="${top}" x2="${left + 20}" y2="${top}"/>` : ""
  const cls = `node fam-${node.family}${taps ? " tapped" : ""}${node.isRoot ? " root" : ""}`
  return `<g class="${cls}" data-id="${node.id}">
<circle class="pip" cx="${x}" cy="${y}" r="2.5"/>
<g class="body"><rect class="outer" x="${left}" y="${top}" width="${width}" height="${height}" rx="2"/>
<rect class="inner" x="${left + 3}" y="${top + 3}" width="${width - 6}" height="${height - 6}" rx="1"/>
<use class="glyph" href="#g-${node.family}" x="${left + 10}" y="${top + 11}" width="8" height="8"/>
<text class="op" x="${left + 24}" y="${top + 18.5}">${escape(node.label)}</text>
<text class="detail" x="${left + 10}" y="${top + 31}">${escape(clip(node.detail, width))}</text>${chips}
</g><g class="node-chip" data-node-chip="${node.id}"><rect rx="1"/><text x="${x}" y="${y}"><tspan></tspan><tspan class="chip-tap"></tspan></text></g>${tapTag}${tick}${
    ports.in ? `<circle class="port" cx="${x}" cy="${top}" r="2"/>` : ""
  }${ports.out ? `<circle class="port" cx="${x}" cy="${top + height}" r="2"/>` : ""}</g>`
}

/** Labels for the dot level, drawn above all nodes; tapped and loss nodes keep theirs as landmarks. */
const renderLabel = ({ node, x, y }: Placed) => {
  const taps = escape(node.taps.join(", ").toUpperCase())
  const name = `${escape(node.label)}${taps ? ` · ${taps}` : ""}`
  const landmark = node.taps.length > 0 || node.family === "loss"
  return `<g class="label${landmark ? " landmark" : ""}${taps ? " tapped" : ""}" data-label="${node.id}">` +
    `<text class="pip-label" x="${x}" y="${y}">${name}</text></g>`
}

/** Op label cut to what fits `roomPx` screen px with chip padding; empty when under two characters fit. */
export const fitLabel = (label: string, roomPx: number) => {
  const max = Math.floor((Math.min(roomPx, CHIP_MAX_PX) - 2 * CHIP_PAD_PX) / FIT_CHAR_PX)
  if (max < 2) return ""
  return label.length <= max ? label : `${label.slice(0, max - 1)}…`
}

interface Chip {
  readonly rect: SVGRectElement
  readonly text: SVGTextElement
  readonly op: SVGTSpanElement
  readonly tap: SVGTSpanElement
  readonly place: Placed
}

/**
 * Sizes a chip in graph units so it draws at a fixed screen size: label width plus padding, box height or more.
 * A tapped node's chip reads `op TAP`, the tap name in the accent. Chips near the side stay inside `extent`.
 */
const paintChip = ({ rect, text, op, tap, place }: Chip, s: number, extent: number) => {
  const { label: name, taps } = place.node
  const full = taps.length > 0 ? `${name} ${taps.join(",").toUpperCase()}` : name
  const label = fitLabel(full, place.room * s - CHIP_GAP_PX)
  const head = label.slice(0, name.length)
  const rest = label.slice(name.length)
  if (op.textContent !== head) op.textContent = head
  if (tap.textContent !== rest) tap.textContent = rest
  const w = (label.length === 0 ? 8 : label.length * FIT_CHAR_PX + 2 * CHIP_PAD_PX) / s
  const h = label.length === 0 ? w : Math.max(CHIP_H_PX / s, place.height)
  const x = Math.min(Math.max(place.x, w / 2), extent - w / 2)
  text.setAttribute("x", String(x))
  rect.setAttribute("x", String(x - w / 2))
  rect.setAttribute("y", String(place.y - h / 2))
  rect.setAttribute("width", String(w))
  rect.setAttribute("height", String(h))
}

const polyLength = (points: ReadonlyArray<Point>) =>
  points.reduce((sum, p, i) => (i === 0 ? 0 : sum + Math.hypot(p.x - points[i - 1]!.x, p.y - points[i - 1]!.y)), 0)

/** Long edges fade so the cross-graph runs to the optimizer and gradients stay in the background. */
const renderEdges = (edges: Laid["edges"]) => {
  const lengths = edges.map((e) => polyLength(e.points))
  const longest = Math.max(1, ...lengths)
  return edges.map(({ points, count }, i) => {
    if (points.length === 0) return ""
    const mid = points[Math.floor(points.length / 2)]!
    const label = count > 1 ? `<text class="mult" x="${mid.x + 5}" y="${mid.y}">×${count}</text>` : ""
    const k = (lengths[i] ?? 0) / longest
    const alpha = EDGE_ALPHA.short + (EDGE_ALPHA.long - EDGE_ALPHA.short) * k
    return `<path data-edge="${i}" d="${smoothPath(points)}" stroke-opacity="${alpha.toFixed(3)}"/>${label}`
  }).join("")
}

const pulseEdges = (laid: Laid, view: ViewGraph): Array<PulseEdge> => {
  const family = new Map(view.nodes.map((n) => [n.id, n.family]))
  const windows = edgeWindows(nodeDepths(view.nodes.map((n) => n.id), view.edges), view.edges)
  return laid.edges.flatMap((e, i) =>
    e.points.length < 2 ? [] : [{
      poly: samplePolyline(e.points),
      window: windows[i] ?? [0, 1],
      accent: family.get(e.to) === "backward" || family.get(e.to) === "optimizer",
      lossEntry: family.get(e.to) === "loss"
    }]
  )
}

export const makeGraphView = ({ root, meta }: { root: HTMLElement; meta: HTMLElement }) => {
  root.innerHTML = `
<svg class="glyph-defs" aria-hidden="true"><defs>${glyphSymbols()}
<linearGradient id="wash" x1="0" x2="1"><stop offset="0" stop-color="currentColor" stop-opacity="0.02"/><stop offset="1" stop-color="currentColor" stop-opacity="0.2"/></linearGradient></defs></svg>
<div class="graph-toolbar">
  <button class="tag" data-tab="forward">forward</button><button class="tag" data-tab="train">train</button>
  <label class="tag check"><input type="checkbox" checked data-hide-scalars><span class="mark"></span>hide scalars</label>
  <span class="spacer"></span>
  <span class="graph-step" aria-hidden="true"></span>
  <button class="tag" data-zoom="out" title="Zoom out (ctrl + wheel)">−</button><button class="tag" data-zoom="in" title="Zoom in (ctrl + wheel)">+</button>
  <button class="tag" data-fit>fit</button>
</div>
<div class="graph-legend">${
    families.map((f) => `<span><svg width="8" height="8"><use href="#g-${f}"/></svg>${f}</span>`).join("")
  }<span class="tap-key"><i></i>tapped</span></div>
<div class="graph-canvas box"><div class="graph-empty">waiting for graph…</div><div class="graph-scroll" hidden><i></i></div></div>
<div class="graph-tooltip box" hidden></div>`
  const canvas = root.querySelector<HTMLElement>(".graph-canvas")!
  const tooltip = root.querySelector<HTMLElement>(".graph-tooltip")!
  const hideScalarsBox = root.querySelector<HTMLInputElement>("[data-hide-scalars]")!
  const scrollTrack = root.querySelector<HTMLElement>(".graph-scroll")!
  const scrollThumb = scrollTrack.querySelector<HTMLElement>("i")!

  const svg = document.createElementNS(SVG_NS, "svg")
  let viewBox: ViewBox = { x: 0, y: 0, w: 100, h: 100 }
  let bounds = { width: 100, height: 100 }
  let size = { width: 1, height: 1 }
  let autoFit = true
  let lod = ""
  let zoom = 0
  const stepHud = root.querySelector<HTMLElement>(".graph-step")!
  const showStep = (step: number) => {
    stepHud.textContent = `step ${step.toLocaleString("en-US")}`
    stepHud.classList.remove("beat")
    void stepHud.offsetWidth
    stepHud.classList.add("beat")
  }
  const pulses = makePulses({
    host: canvas,
    viewBox: () => viewBox,
    onWave: (step) => {
      showStep(step)
      onWave()
    }
  })
  let onWave = () => {}

  let chips: Array<Chip> = []
  const paintChips = (s: number) => {
    for (const c of chips) paintChip(c, s, bounds.width)
  }
  const applyLod = () => {
    const { s } = screenTransform(viewBox, size.width, size.height)
    const next = LOD.find((l) => s >= l.min)!.cls
    if (s !== zoom) {
      zoom = s
      svg.style.setProperty("--s", s.toFixed(4))
      if (next === "lod-fit") paintChips(s)
    }
    if (next === lod) return
    if (lod) svg.classList.remove(lod)
    svg.classList.add(next)
    lod = next
  }
  const paintScroll = () => {
    const { s, ty } = screenTransform(viewBox, size.width, size.height)
    const full = bounds.height * s
    const tall = full > size.height + 1
    scrollTrack.hidden = !tall
    if (!tall) return
    scrollThumb.style.top = `${(Math.max(0, -ty) / full) * 100}%`
    scrollThumb.style.height = `${Math.min(1, size.height / full) * 100}%`
  }
  const setViewBox = (b: ViewBox, fitted = false) => {
    autoFit = fitted
    viewBox = b
    svg.setAttribute("viewBox", `${b.x} ${b.y} ${b.w} ${b.h}`)
    applyLod()
    paintScroll()
    pulses.invalidate()
  }
  const fit = () => setViewBox(fitView({ graph: bounds, view: size }), true)
  new ResizeObserver(([entry]) => {
    if (!entry) return
    size = { width: entry.contentRect.width || 1, height: entry.contentRect.height || 1 }
    if (autoFit) fit()
    else applyLod()
  }).observe(svg)

  const graphs = new Map<GraphId, GraphMessage>()
  let active: GraphId | undefined
  let userPicked = false
  let view: ViewGraph = { nodes: [], edges: [] }
  let nodesById = new Map<number, ViewNode>()
  let selected: number | undefined

  const select = (id: number | undefined) => {
    selected = id
    const path = id === undefined ? undefined : activePath(id, view.edges)
    svg.classList.toggle("has-selection", path !== undefined)
    for (const el of svg.querySelectorAll<SVGGElement>(".node")) {
      const nid = Number(el.dataset["id"])
      el.classList.toggle("is-active", nid === id)
      el.classList.toggle("on-path", path?.has(nid) ?? false)
    }
    for (const el of svg.querySelectorAll<SVGGElement>("[data-label]")) {
      el.classList.toggle("is-active", Number(el.dataset["label"]) === id)
    }
    for (const el of svg.querySelectorAll<SVGPathElement>("[data-edge]")) {
      const e = view.edges[Number(el.dataset["edge"])]
      el.classList.toggle("on-path", !!e && !!path && path.has(e.from) && path.has(e.to))
    }
  }

  const render = () => {
    const message = active ? graphs.get(active) : undefined
    if (!message) return
    view = collapse({ message, hideScalars: hideScalarsBox.checked })
    const over = view.nodes.length > NODE_BUDGET
    const trainNodes = graphs.get("train")?.snapshot.nodes.length ?? 0
    const trainOver = active === "forward" && trainNodes > NODE_BUDGET
    meta.textContent = `${view.nodes.length} nodes · ${view.edges.length} edges` +
      (over ? ` · over ${NODE_BUDGET} node budget, layout may be slow` : "") +
      (trainOver ? ` · train graph has ${trainNodes} nodes` : "")
    meta.classList.toggle("warn", over || trainOver)
    const laid = layout(view)
    nodesById = new Map(view.nodes.map((n) => [n.id, n]))
    const hasIn = new Set(view.edges.map((e) => e.to))
    const hasOut = new Set(view.edges.map((e) => e.from))
    svg.innerHTML = `<g class="edges">${renderEdges(laid.edges)}</g><g class="nodes">${
      laid.placed.map((p) => renderNode(p, { in: hasIn.has(p.node.id), out: hasOut.has(p.node.id) })).join("")
    }</g><g class="labels">${laid.placed.map(renderLabel).join("")}</g>`
    const places = new Map(laid.placed.map((p) => [p.node.id, p]))
    chips = [...svg.querySelectorAll<SVGGElement>("[data-node-chip]")].flatMap((el) => {
      const place = places.get(Number(el.dataset["nodeChip"]))
      const rect = el.querySelector("rect")
      const text = el.querySelector("text")
      const [op, tap] = el.querySelectorAll("tspan")
      return place && rect && text && op && tap ? [{ rect, text, op, tap, place }] : []
    })
    zoom = 0
    bounds = { width: laid.width, height: laid.height }
    if (!svg.isConnected) canvas.replaceChildren(svg, ...canvas.querySelectorAll(".graph-pulses, .graph-scroll"))
    pulses.setLayout({ edges: pulseEdges(laid, view), mode: active ?? "train" })
    fit()
    select(selected !== undefined && nodesById.has(selected) ? selected : undefined)
    for (const b of root.querySelectorAll<HTMLButtonElement>("[data-tab]")) {
      b.classList.toggle("active", b.dataset["tab"] === active)
      b.disabled = !graphs.has(b.dataset["tab"] === "train" ? "train" : "forward")
    }
  }

  root.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((b) =>
    b.addEventListener("click", () => {
      active = b.dataset["tab"] === "train" ? "train" : "forward"
      userPicked = true
      render()
    })
  )
  root.querySelector("[data-fit]")!.addEventListener("click", fit)
  root.querySelectorAll<HTMLButtonElement>("[data-zoom]").forEach((b) =>
    b.addEventListener("click", () => {
      const k = b.dataset["zoom"] === "in" ? 1 / ZOOM_STEP : ZOOM_STEP
      setViewBox(zoomAt(viewBox, k, { x: viewBox.x + viewBox.w / 2, y: viewBox.y + viewBox.h / 2 }))
    })
  )
  hideScalarsBox.addEventListener("change", render)

  attachPanZoom({
    svg,
    box: () => viewBox,
    bounds: () => bounds,
    set: (b) => setViewBox(b),
    onTap: (target) => {
      const id = target?.closest("[data-id]")?.getAttribute("data-id")
      select(id === null || id === undefined || Number(id) === selected ? undefined : Number(id))
    }
  })
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && selected !== undefined) select(undefined)
  })

  let hovered: Element | null = null
  const hover = (id: number | undefined) => {
    const next = id === undefined ? null : svg.querySelector(`[data-label="${id}"]`)
    if (next === hovered) return
    hovered?.classList.remove("is-hover")
    next?.classList.add("is-hover")
    hovered = next
  }
  svg.addEventListener("mousemove", (e) => {
    const target = e.target instanceof Element ? e.target.closest("[data-id]") : null
    const node = target ? nodesById.get(Number(target.getAttribute("data-id"))) : undefined
    hover(node?.id)
    tooltip.hidden = !node
    if (!node) return
    tooltip.innerHTML = renderTooltip(node)
    const r = root.getBoundingClientRect()
    tooltip.style.left = `${Math.min(e.clientX - r.left + 14, r.width - 300)}px`
    tooltip.style.top = `${e.clientY - r.top + 14}px`
  })
  svg.addEventListener("mouseleave", () => {
    tooltip.hidden = true
    hover(undefined)
  })

  const defaultTab = (): GraphId => {
    const train = graphs.get("train")
    if (!train || (graphs.has("forward") && train.snapshot.nodes.length > NODE_BUDGET)) return "forward"
    return "train"
  }

  return {
    setGraph: (message: GraphMessage) => {
      graphs.set(message.graphId, message)
      const next = userPicked && active && graphs.has(active) ? active : defaultTab()
      if (next === active && next !== message.graphId) return
      active = next
      render()
    },
    pulses,
    onWave: (listener: () => void) => void (onWave = listener)
  }
}
