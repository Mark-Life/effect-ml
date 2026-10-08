import type { GraphMessage, GraphSnapshotNode } from "../protocol.ts"
import { type Family, familyOf } from "./op-family.ts"

export interface ViewNode {
  readonly id: number
  readonly op: string
  readonly label: string
  readonly detail: string
  readonly family: Family
  readonly taps: ReadonlyArray<string>
  readonly chips: ReadonlyArray<string>
  readonly isRoot: boolean
  readonly source: GraphSnapshotNode
}

export interface ViewEdge {
  readonly from: number
  readonly to: number
  /** Operand multiplicity, e.g. 2 for `add(a, a)`. */
  readonly count: number
}

export interface ViewGraph {
  readonly nodes: ReadonlyArray<ViewNode>
  readonly edges: ReadonlyArray<ViewEdge>
}

const SCALAR_CONSTANTS = new Set(["full", "zeros", "ones"])

const isHiddenScalar = (node: GraphSnapshotNode) => SCALAR_CONSTANTS.has(node.op) && node.shape.length === 0

const formatNumber = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toPrecision(3)) : String(v)

const scalarValue = (node: GraphSnapshotNode) =>
  node.op === "zeros" ? "0" : node.op === "ones" ? "1" : formatNumber(node.attributes["value"])

const labelOf = (node: GraphSnapshotNode, slotLabels: ReadonlyArray<string>) => {
  const slot = node.attributes["slot"]
  if ((node.op === "input" || node.op === "scalar_input") && typeof slot === "number") {
    return slotLabels[slot] ?? `${node.op}[${slot}]`
  }
  return node.op
}

export const detailOf = (node: Pick<GraphSnapshotNode, "shape" | "dtype">) => `[${node.shape.join(",")}] ${node.dtype}`

/**
 * Folds `expose` nodes into their input (which gets the tap name) and,
 * when `hideScalars`, drops rank-0 constants and shows their value as a chip on consumers.
 */
export const collapse = (
  { message, hideScalars }: { message: Pick<GraphMessage, "snapshot" | "slotLabels">; hideScalars: boolean }
): ViewGraph => {
  const { nodes, roots } = message.snapshot
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const resolve = (id: number): number => {
    let current = id
    for (let node = byId.get(current); node?.op === "expose" && node.inputs[0] !== undefined; node = byId.get(current)) {
      current = node.inputs[0]
    }
    return current
  }

  const taps = new Map<number, Array<string>>()
  for (const node of nodes) {
    if (node.op !== "expose") continue
    const target = resolve(node.id)
    taps.set(target, [...(taps.get(target) ?? []), node.exposeName ?? String(node.attributes["name"] ?? "")])
  }

  const rootIds = new Set(roots.map(resolve))
  const hidden = (node: GraphSnapshotNode) => node.op === "expose" || (hideScalars && isHiddenScalar(node))

  const edgeCounts = new Map<string, ViewEdge>()
  const viewNodes = nodes.filter((n) => !hidden(n)).map((node): ViewNode => {
    const chips: Array<string> = []
    for (const input of node.inputs) {
      const from = resolve(input)
      const source = byId.get(from)
      if (source && hideScalars && isHiddenScalar(source)) {
        chips.push(scalarValue(source))
        continue
      }
      const key = `${from}->${node.id}`
      const prev = edgeCounts.get(key)
      edgeCounts.set(key, { from, to: node.id, count: (prev?.count ?? 0) + 1 })
    }
    return {
      id: node.id,
      op: node.op,
      label: labelOf(node, message.slotLabels),
      detail: detailOf(node),
      family: familyOf(node.op),
      taps: taps.get(node.id) ?? [],
      chips,
      isRoot: rootIds.has(node.id),
      source: node
    }
  })
  return { nodes: viewNodes, edges: [...edgeCounts.values()] }
}

/** Nodes reachable from `roots`, keeping snapshot postorder. */
export const prune = (
  snapshot: GraphMessage["snapshot"],
  roots: ReadonlyArray<number>
): GraphMessage["snapshot"] => {
  const byId = new Map(snapshot.nodes.map((n) => [n.id, n]))
  const keep = new Set<number>()
  const stack = [...roots]
  while (stack.length > 0) {
    const id = stack.pop()!
    if (keep.has(id)) continue
    keep.add(id)
    stack.push(...(byId.get(id)?.inputs ?? []))
  }
  return { nodes: snapshot.nodes.filter((n) => keep.has(n.id)), roots }
}
