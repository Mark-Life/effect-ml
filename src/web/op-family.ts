export const families = [
  "io",
  "const",
  "elementwise",
  "reduce",
  "matmul",
  "attention",
  "norm",
  "shape",
  "index",
  "loss",
  "backward",
  "optimizer",
  "meta",
  "other"
] as const

export type Family = typeof families[number]

const table: Record<Exclude<Family, "backward" | "other">, ReadonlyArray<string>> = {
  io: ["leaf", "input", "scalar_input", "from_bytes"],
  const: ["zeros", "ones", "full", "randn", "uniform", "arange", "eye"],
  elementwise: [
    "add",
    "sub",
    "mul",
    "div",
    "eq",
    "gt",
    "lt",
    "ge",
    "le",
    "maximum",
    "minimum",
    "neg",
    "abs",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tanh",
    "relu",
    "erf",
    "gelu",
    "floor",
    "ceil",
    "round",
    "sign",
    "where",
    "pow",
    "cast"
  ],
  reduce: ["sum", "mean", "max", "min", "prod", "argmax", "argmin", "cumsum", "topKIndices"],
  matmul: [
    "matmul",
    "linear",
    "quantized_linear",
    "expertLinearRows",
    "groupedExpertLinearRows",
    "conv1d",
    "conv2d",
    "conv_transpose1d",
    "conv_transpose2d",
    "short_conv1d"
  ],
  attention: [
    "sdpa",
    "sdpa_configured",
    "kv_attention",
    "kda_chunk",
    "kda_recurrence",
    "position_embedding",
    "rotary_embedding",
    "rotary_embedding_explicit"
  ],
  norm: ["layer_norm", "rms_norm"],
  shape: ["reshape", "permute", "slice", "concat", "broadcast_to", "last_token_row"],
  index: ["index_select", "gather", "scatter_add", "quantized_embedding"],
  loss: ["cross_entropy", "chunked_head_ce"],
  optimizer: ["adam_w_step", "adam_w_out", "sgd_step", "sgd_out"],
  meta: ["expose", "stop_gradient", "checkpoint"]
}

const byOp = new Map<string, Family>(
  families.flatMap((family) =>
    family === "backward" || family === "other" ? [] : table[family].map((op) => [op, family] as const)
  )
)

export const familyOf = (op: string): Family => {
  const direct = byOp.get(op)
  if (direct === "optimizer") return direct
  if (op.includes("backward") || op.endsWith("_out")) return "backward"
  return direct ?? "other"
}

/** SVG markup on an 8x8 box; strokes and fills use `currentColor`. */
export const familyGlyph: Record<Family, string> = {
  io: `<circle cx="4" cy="4" r="3.2"/><path d="M0.8 4h6.4M4 0.8v6.4"/>`,
  const: `<rect x="2" y="2" width="4" height="4"/>`,
  elementwise: `<circle cx="4" cy="4" r="1.5" fill="currentColor" stroke="none"/>`,
  reduce: `<path d="M1 1.5h6L4 6.8z"/>`,
  matmul: `<path d="M1 1h6v6H1zM4 1v6M1 4h6"/>`,
  attention: `<path d="M0.8 7Q4 -1 7.2 7M0.8 1Q4 9 7.2 1"/>`,
  norm: `<path d="M1 4h6M1 2v4M7 2v4"/>`,
  shape: `<path d="M1 3.5V1h2.5M4.5 7H7V4.5"/>`,
  index: `<path d="M3.5 1H7v6H3.5M0.5 4h4.5M3.5 2.5L5 4L3.5 5.5"/>`,
  loss: `<path d="M4 0.5L7.5 4L4 7.5L0.5 4z" fill="currentColor"/>`,
  backward: `<path d="M4 1.5L1.5 4L4 6.5M7 1.5L4.5 4L7 6.5"/>`,
  optimizer: `<path d="M1 7h2V5h2V3h2V1"/>`,
  meta: `<circle cx="4" cy="4" r="3" stroke-dasharray="1.6 1.4"/>`,
  other: `<circle cx="4" cy="4" r="2"/>`
}

/** `<symbol id="g-<family>">` for each family, to place with `<use href="#g-<family>">`. */
export const glyphSymbols = () =>
  families.map((f) =>
    `<symbol id="g-${f}" viewBox="0 0 8 8" fill="none" stroke="currentColor" stroke-width="1">${familyGlyph[f]}</symbol>`
  ).join("")
