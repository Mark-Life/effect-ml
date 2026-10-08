// Generates src/web/fixtures/graph.json: a hand-shaped GraphSnapshot for the viewer fixture.
import type { GraphSnapshot, GraphSnapshotNode } from "../../protocol.ts"
import { prune } from "../graph-model.ts"

const nodes: Array<GraphSnapshotNode> = []
let nextId = 100
const n = (
  op: string,
  shape: Array<number>,
  inputs: Array<number> = [],
  attributes: Record<string, unknown> = {},
  dtype = "f32"
) => {
  const id = nextId++
  nodes.push({ id, op, shape, dtype, inputs, attributes })
  return id
}
const expose = (a: number, name: string) => {
  const src = nodes.find((x) => x.id === a)!
  const id = nextId++
  nodes.push({ id, op: "expose", shape: src.shape, dtype: src.dtype, inputs: [a], attributes: { name }, exposeName: name })
  return id
}

const B = 3831
const slotLabels = ["embed.W_E", "embed.W_pos", "mlp.W_in", "mlp.b_in", "mlp.W_out", "unembed.W_U", "input", "target"]
const W_E = n("input", [114, 128], [], { slot: 0 })
const W_pos = n("input", [3, 128], [], { slot: 1 })
const W_in = n("input", [128, 512], [], { slot: 2 })
const b_in = n("input", [512], [], { slot: 3 })
const W_out = n("input", [512, 128], [], { slot: 4 })
const W_U = n("input", [128, 114], [], { slot: 5 })
const x = n("input", [B, 3], [], { slot: 6 }, "i64")
const y = n("input", [B], [], { slot: 7 }, "i64")

const flatX = n("reshape", [B * 3], [x], { shape: [B * 3] }, "i64")
const emb = n("index_select", [B * 3, 128], [W_E, flatX], { dim: 0 })
const emb3 = n("reshape", [B, 3, 128], [emb], { shape: [B, 3, 128] })
const pos = n("broadcast_to", [B, 3, 128], [W_pos], { shape: [B, 3, 128] })
const h0 = n("add", [B, 3, 128], [emb3, pos])
const h0e = expose(h0, "embed")
const q = n("reshape", [B, 1, 3, 128], [h0e], { shape: [B, 1, 3, 128] })
const att = n("sdpa_configured", [B, 1, 3, 128], [q, q, q], { scale: 0.0884, causal: true })
const att3 = n("reshape", [B, 3, 128], [att], { shape: [B, 3, 128] })
const h1 = n("add", [B, 3, 128], [h0e, att3])
const pre = n("add", [B, 3, 512], [n("matmul", [B, 3, 512], [h1, W_in]), b_in])
const hid = n("relu", [B, 3, 512], [pre])
const hidLast = n("reshape", [B, 512], [n("slice", [B, 1, 512], [hid], { ranges: [[0, B, 1], [2, 3, 1], [0, 512, 1]] })], {
  shape: [B, 512]
})
const hidE = expose(hidLast, "mlp.hidden")
const h1Last = n("reshape", [B, 128], [n("slice", [B, 1, 128], [h1], { ranges: [[0, B, 1], [2, 3, 1], [0, 128, 1]] })], {
  shape: [B, 128]
})
const mlpOut = n("matmul", [B, 128], [hidE, W_out])
const h2 = expose(n("add", [B, 128], [h1Last, mlpOut]), "resid.final")
const ln = n("layer_norm", [B, 128], [h2], { eps: 1e-5 })
const logits = expose(n("matmul", [B, 114], [ln, W_U]), "logits")
const ce = n("cross_entropy", [B], [logits, y], { ignoreIndex: -100 })
const invB = n("full", [], [], { value: 1 / B })
const loss = n("mul", [], [n("sum", [], [ce], { dims: [0], keepdims: false }), invB])
const pred = n("argmax", [B], [logits], { dim: 1 }, "i64")
const acc = n("mean", [], [n("cast", [B], [n("eq", [B], [pred, y], {}, "u8")], { dtype: "f32" })], {
  dims: [0],
  keepdims: false
})

// backward
const one = n("ones", [])
const dLogits = n("cross_entropy_backward", [B, 114], [logits, y, n("mul", [], [one, invB])])
const dW_U = n("matmul", [128, 114], [n("permute", [128, B], [ln], { dims: [1, 0] }), dLogits])
const dLn = n("matmul", [B, 128], [dLogits, n("permute", [114, 128], [W_U], { dims: [1, 0] })])
const dH2 = n("layer_norm_backward_out", [B, 128], [dLn, h2], { eps: 1e-5 })
const dW_out = n("matmul", [512, 128], [n("permute", [512, B], [hidLast], { dims: [1, 0] }), dH2])
const dHidLast = n("matmul", [B, 512], [dH2, n("permute", [128, 512], [W_out], { dims: [1, 0] })])
const zero = n("zeros", [])
const mask = n("cast", [B, 3, 512], [n("gt", [B, 3, 512], [pre, zero], {}, "u8")], { dtype: "f32" })
const dHid = n("mul", [B, 3, 512], [n("broadcast_to", [B, 3, 512], [dHidLast], { shape: [B, 3, 512] }), mask])
const db_in = n("sum", [512], [dHid], { dims: [0, 1], keepdims: false })
const dW_in = n("matmul", [128, 512], [
  n("reshape", [128, B * 3], [n("permute", [128, B, 3], [h1], { dims: [2, 0, 1] })], { shape: [128, B * 3] }),
  n("reshape", [B * 3, 512], [dHid], { shape: [B * 3, 512] })
])
const dH1 = n("add", [B, 3, 128], [n("matmul", [B, 3, 128], [dHid, n("permute", [512, 128], [W_in], { dims: [1, 0] })]), dH2])
const dH1x2 = n("add", [B, 3, 128], [dH1, dH1])
const dAtt = n("sdpa_backward", [B, 1, 3, 128], [q, q, q, n("reshape", [B, 1, 3, 128], [dH1x2], { shape: [B, 1, 3, 128] })], {
  scale: 0.0884,
  causal: true
})
const dH0 = n("add", [B, 3, 128], [dH1x2, n("reshape", [B, 3, 128], [dAtt], { shape: [B, 3, 128] })])
const dW_pos = n("sum", [3, 128], [dH0], { dims: [0], keepdims: false })
const dW_E = n("scatter_add", [114, 128], [
  n("zeros", [114, 128]),
  flatX,
  n("reshape", [B * 3, 128], [dH0], { shape: [B * 3, 128] })
], { dim: 0 })
const stop = n("stop_gradient", [114, 128], [dW_E])
const step = n("adam_w_step", [128, 114], [W_U, dW_U], { beta1: 0.9, beta2: 0.98, eps: 1e-8, weightDecay: 1 })
const newW_U = n("adam_w_out", [128, 114], [step])

const train: GraphSnapshot = { nodes, roots: [loss, acc, stop, dW_pos, dW_in, db_in, dW_out, dW_U, newW_U] }
const forward = prune(train, [logits, loss, acc])

await Bun.write(
  new URL("./graph.json", import.meta.url),
  JSON.stringify({ slotLabels, forward, train }, null, 1)
)
console.log(`${nodes.length} nodes`)
