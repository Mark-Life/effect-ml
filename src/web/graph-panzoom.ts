export interface ViewBox {
  readonly x: number
  readonly y: number
  readonly w: number
  readonly h: number
}

interface Size {
  readonly width: number
  readonly height: number
}

const TAP_SLOP = 4
/** Fit never zooms past this; small graphs stay at a calm size. */
const MAX_FIT = 1.25
/** Below this scale 10.5 px labels drop under ~9 screen px; boxes give way to dots. */
export const MIN_LEGIBLE = 0.85
/** Below this chips get too crowded to read; the default view never goes lower, the user can. */
export const MIN_FIT = 0.2

/** Screen px per graph unit and the screen offset of the viewBox origin (`preserveAspectRatio` meet, centered). */
export const screenTransform = (box: ViewBox, width: number, height: number) => {
  const s = Math.min(width / box.w, height / box.h)
  return { s, tx: (width - box.w * s) / 2 - box.x * s, ty: (height - box.h * s) / 2 - box.y * s }
}

/**
 * Default view. Whole graph when it fits at a legible scale; else fit the width (no lower than `MIN_FIT`, a wider
 * graph is centered and pans), start at the top and let the graph scroll. Level of detail then picks boxes or
 * labelled chips from the resulting scale.
 */
export const fitView = ({ graph, view }: { graph: Size; view: Size }): ViewBox => {
  const all = Math.min(view.width / graph.width, view.height / graph.height)
  const s = Math.min(all >= MIN_LEGIBLE ? all : Math.max(MIN_FIT, view.width / graph.width), MAX_FIT)
  const w = view.width / s
  const h = view.height / s
  return { x: (graph.width - w) / 2, y: h < graph.height ? 0 : (graph.height - h) / 2, w, h }
}

/** Zooms `b` by factor `k` (>1 zooms out) around graph point `p`. */
export const zoomAt = (b: ViewBox, k: number, p: { x: number; y: number }): ViewBox => ({
  x: p.x - (p.x - b.x) * k,
  y: p.y - (p.y - b.y) * k,
  w: b.w * k,
  h: b.h * k
})

const clampAxis = (pos: number, span: number, extent: number) =>
  span >= extent ? pos : Math.min(Math.max(pos, 0), extent - span)

/**
 * Wheel scrolls the graph (the page takes over at the graph's edge), ctrl/cmd + wheel or pinch zooms around the
 * cursor, drag pans. A press that barely moves is reported as a tap.
 */
export const attachPanZoom = (
  { svg, box, bounds, set, onTap }: {
    svg: SVGSVGElement
    box: () => ViewBox
    bounds: () => Size
    set: (b: ViewBox) => void
    onTap: (target: Element | null) => void
  }
) => {
  const toSvg = (clientX: number, clientY: number) => {
    const r = svg.getBoundingClientRect()
    const b = box()
    const s = Math.max(b.w / r.width, b.h / r.height)
    return {
      s,
      x: b.x + (clientX - r.left) * s - (r.width * s - b.w) / 2,
      y: b.y + (clientY - r.top) * s - (r.height * s - b.h) / 2
    }
  }
  svg.addEventListener("wheel", (e) => {
    const b = box()
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault()
      set(zoomAt(b, Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), toSvg(e.clientX, e.clientY)))
      return
    }
    const { s } = toSvg(e.clientX, e.clientY)
    const { width, height } = bounds()
    const unit = e.deltaMode === 1 ? 16 : 1
    const x = clampAxis(b.x + e.deltaX * unit * s, b.w, width)
    const y = clampAxis(b.y + e.deltaY * unit * s, b.h, height)
    if (Math.abs(x - b.x) < 0.01 && Math.abs(y - b.y) < 0.01) return
    e.preventDefault()
    set({ ...b, x, y })
  }, { passive: false })
  let drag: { x: number; y: number; box: ViewBox; s: number; target: Element | null; moved: boolean } | undefined
  svg.addEventListener("pointerdown", (e) => {
    drag = {
      x: e.clientX,
      y: e.clientY,
      box: box(),
      s: toSvg(e.clientX, e.clientY).s,
      target: e.target instanceof Element ? e.target : null,
      moved: false
    }
    svg.setPointerCapture(e.pointerId)
  })
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return
    const dx = e.clientX - drag.x
    const dy = e.clientY - drag.y
    if (!drag.moved && Math.hypot(dx, dy) < TAP_SLOP) return
    drag.moved = true
    set({ ...drag.box, x: drag.box.x - dx * drag.s, y: drag.box.y - dy * drag.s })
  })
  svg.addEventListener("pointerup", () => {
    if (drag && !drag.moved) onTap(drag.target)
    drag = undefined
  })
}
