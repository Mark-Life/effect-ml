import type { ViewNode } from "./graph-model.ts"

const esc = (s: string) => s.replace(/[&<>]/g, (c) => (c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;"))

/** Key/value grid for a node, then its attributes as compact JSON. */
export const renderTooltip = (node: ViewNode) => {
  const rows: ReadonlyArray<readonly [string, string]> = [
    ["id", String(node.id)],
    ["op", node.op],
    ["shape", `[${node.source.shape.join(", ")}]`],
    ["dtype", node.source.dtype],
    ["inputs", node.source.inputs.length > 0 ? node.source.inputs.join(", ") : "none"],
    ...(node.taps.length > 0 ? [["taps", node.taps.join(", ")] as const] : [])
  ]
  const attributes = JSON.stringify(node.source.attributes)
  return `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>${
    attributes === "{}" ? "" : `<div class="tooltip-attrs"><span>attributes</span>${esc(attributes)}</div>`
  }`
}
