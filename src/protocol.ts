import { Schema } from "effect"

const Shape = Schema.Array(Schema.Finite)

export const GraphSnapshotNode = Schema.Struct({
  id: Schema.Finite,
  op: Schema.String,
  shape: Shape,
  dtype: Schema.String,
  inputs: Schema.Array(Schema.Finite),
  attributes: Schema.Record(Schema.String, Schema.Unknown),
  exposeName: Schema.optionalKey(Schema.String)
})
export interface GraphSnapshotNode extends Schema.Schema.Type<typeof GraphSnapshotNode> {}

export const GraphSnapshot = Schema.Struct({
  nodes: Schema.Array(GraphSnapshotNode),
  roots: Schema.Array(Schema.Finite)
})
export interface GraphSnapshot extends Schema.Schema.Type<typeof GraphSnapshot> {}

export const InstructionDiagnostics = Schema.Struct({ kind: Schema.String, count: Schema.Finite })

export const CompilePhaseDiagnostics = Schema.Struct({ phase: Schema.String, nanoseconds: Schema.Finite })

export const MemoryDiagnostics = Schema.Struct({
  externalBytes: Schema.Finite,
  persistentBytes: Schema.Finite,
  stateBytes: Schema.Finite,
  outputBytes: Schema.Finite,
  workspaceBytes: Schema.Finite,
  transactionBytes: Schema.Finite,
  peakLiveBytes: Schema.Finite,
  packingOverheadBytes: Schema.Finite
})
export interface MemoryDiagnostics extends Schema.Schema.Type<typeof MemoryDiagnostics> {}

export const LegalizationDiagnostics = Schema.Struct({
  targetBackend: Schema.String,
  targetArchitecture: Schema.String,
  loweringAbiRevision: Schema.Finite,
  policyRevision: Schema.Finite,
  capabilityQueries: Schema.Finite,
  nativeLoweringUnits: Schema.Finite,
  legalizedLoweringUnits: Schema.Finite,
  kernelLocalLegalizations: Schema.Finite,
  materializedConversions: Schema.Finite,
  materializedConversionBytes: Schema.Finite,
  decompositions: Schema.Finite,
  rejectedRegionCandidates: Schema.Finite
})

/** Mirrors `Runtime.ExecutableDiagnostics` from effect-torch core. */
export const ExecutableDiagnostics = Schema.Struct({
  semanticNodesBeforeOptimization: Schema.Finite,
  semanticNodesAfterOptimization: Schema.Finite,
  instructions: Schema.Array(InstructionDiagnostics),
  pipelineCount: Schema.Finite,
  commandCount: Schema.Finite,
  synchronizationCount: Schema.Finite,
  memory: MemoryDiagnostics,
  legalization: Schema.optional(LegalizationDiagnostics),
  compilePhases: Schema.optionalKey(Schema.Array(CompilePhaseDiagnostics))
})
export interface ExecutableDiagnostics extends Schema.Schema.Type<typeof ExecutableDiagnostics> {}

export const RunInfo = Schema.TaggedStruct("RunInfo", {
  runId: Schema.String,
  task: Schema.String,
  backend: Schema.String,
  totalSteps: Schema.Finite,
  probeEvery: Schema.Finite,
  parameters: Schema.Array(Schema.Struct({ name: Schema.String, shape: Shape }))
})
export interface RunInfo extends Schema.Schema.Type<typeof RunInfo> {}

export const GraphId = Schema.Literals(["forward", "train"])
export type GraphId = typeof GraphId.Type

export const GraphMessage = Schema.TaggedStruct("Graph", {
  graphId: GraphId,
  /** Input slot -> label: parameter names, then "input", "target". */
  slotLabels: Schema.Array(Schema.String),
  snapshot: GraphSnapshot
})
export interface GraphMessage extends Schema.Schema.Type<typeof GraphMessage> {}

export const ProgramId = Schema.Literals(["train-step", "probe"])
export type ProgramId = typeof ProgramId.Type

export const DiagnosticsMessage = Schema.TaggedStruct("Diagnostics", {
  programId: ProgramId,
  signature: Schema.String,
  diagnostics: ExecutableDiagnostics
})
export interface DiagnosticsMessage extends Schema.Schema.Type<typeof DiagnosticsMessage> {}

export const MetricPoint = Schema.Struct({
  step: Schema.Finite,
  loss: Schema.Finite,
  stepMs: Schema.Finite,
  memoryBytes: Schema.Finite
})
export interface MetricPoint extends Schema.Schema.Type<typeof MetricPoint> {}

export const Metrics = Schema.TaggedStruct("Metrics", { points: Schema.Array(MetricPoint) })
export interface Metrics extends Schema.Schema.Type<typeof Metrics> {}

export const EvalPoint = Schema.TaggedStruct("Eval", {
  step: Schema.Finite,
  trainLoss: Schema.Finite,
  testLoss: Schema.Finite,
  trainAcc: Schema.Finite,
  testAcc: Schema.Finite,
  probeMs: Schema.Finite
})
export interface EvalPoint extends Schema.Schema.Type<typeof EvalPoint> {}

export const Status = Schema.TaggedStruct("Status", {
  state: Schema.Literals(["compiling", "training", "done", "failed"]),
  message: Schema.optionalKey(Schema.String)
})
export interface Status extends Schema.Schema.Type<typeof Status> {}

export const ServerMessage = Schema.Union([RunInfo, GraphMessage, DiagnosticsMessage, Metrics, EvalPoint, Status])
export type ServerMessage = RunInfo | GraphMessage | DiagnosticsMessage | Metrics | EvalPoint | Status

export const HeatmapHeader = Schema.TaggedStruct("Heatmap", {
  name: Schema.String,
  step: Schema.Finite,
  sourceShape: Shape,
  rows: Schema.Finite,
  cols: Schema.Finite,
  min: Schema.Finite,
  max: Schema.Finite,
  /** `probability`: values in [0, 1] such as attention weights; drawn on a log-odds ramp so both ends spread out. */
  scale: Schema.Literals(["sequential", "diverging", "probability"]),
  axes: Schema.optionalKey(Schema.Struct({ rows: Schema.String, cols: Schema.String })),
  /** For two-level maps: which level is the signal. `low` lights the min cells (wrong answers in `correct`). */
  highlight: Schema.optionalKey(Schema.Literals(["low", "high"])),
  /** Names of the two levels of a two-level map, shown in its legend and hover readout. */
  levels: Schema.optionalKey(Schema.Struct({ low: Schema.String, high: Schema.String })),
  /**
   * The payload carries a second `rows x cols` plane after the values: cells where it is 1 are drawn dimmed and
   * left out of the level shares. `dimmed` and `shown` name the two groups in the legend (e.g. train and test
   * pairs in `correct`).
   */
  mask: Schema.optionalKey(Schema.Struct({ dimmed: Schema.String, shown: Schema.String }))
})
export interface HeatmapHeader extends Schema.Schema.Type<typeof HeatmapHeader> {}

export const HistogramHeader = Schema.TaggedStruct("Histogram", {
  name: Schema.String,
  step: Schema.Finite,
  sourceShape: Shape,
  bins: Schema.Finite,
  min: Schema.Finite,
  max: Schema.Finite,
  nonFinite: Schema.Finite
})
export interface HistogramHeader extends Schema.Schema.Type<typeof HistogramHeader> {}

export const ProbeHeader = Schema.Union([HeatmapHeader, HistogramHeader])
export type ProbeHeader = HeatmapHeader | HistogramHeader

export type ProbeFrame =
  | {
    readonly _tag: "Heatmap"
    readonly header: HeatmapHeader
    readonly payload: Float32Array
    readonly mask: Float32Array | undefined
  }
  | { readonly _tag: "Histogram"; readonly header: HistogramHeader; readonly payload: Uint32Array }

export const decodeServerMessage = Schema.decodeUnknownSync(ServerMessage)
const decodeProbeHeader = Schema.decodeUnknownSync(ProbeHeader)

const align4 = (n: number) => (n + 3) & ~3

/** `[u32 LE headerBytes][UTF-8 JSON header][zero pad to 4][payload]` */
export const encodeProbe = (header: ProbeHeader, payload: Float32Array | Uint32Array) => {
  const json = new TextEncoder().encode(JSON.stringify(header))
  const offset = align4(4 + json.byteLength)
  const frame = new Uint8Array(offset + payload.byteLength)
  new DataView(frame.buffer).setUint32(0, json.byteLength, true)
  frame.set(json, 4)
  frame.set(new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength), offset)
  return frame
}

export const decodeProbe = (frame: ArrayBufferLike): ProbeFrame => {
  if (frame.byteLength < 4) throw new Error(`probe frame too short: ${frame.byteLength} bytes`)
  const headerBytes = new DataView(frame).getUint32(0, true)
  const offset = align4(4 + headerBytes)
  if (offset > frame.byteLength || (frame.byteLength - offset) % 4 !== 0) {
    throw new Error(`probe frame: header of ${headerBytes} bytes does not fit a ${frame.byteLength}-byte frame`)
  }
  const header = decodeProbeHeader(JSON.parse(new TextDecoder().decode(new Uint8Array(frame, 4, headerBytes))))
  const elements = (frame.byteLength - offset) / 4
  const cells = header._tag === "Heatmap" ? header.rows * header.cols : header.bins
  const expected = header._tag === "Heatmap" && header.mask ? cells * 2 : cells
  if (elements !== expected) {
    throw new Error(`probe ${header.name}: payload has ${elements} values, header expects ${expected}`)
  }
  if (header._tag === "Histogram") return { _tag: "Histogram", header, payload: new Uint32Array(frame, offset, elements) }
  return {
    _tag: "Heatmap",
    header,
    payload: new Float32Array(frame, offset, cells),
    mask: header.mask ? new Float32Array(frame, offset + cells * 4, cells) : undefined
  }
}
