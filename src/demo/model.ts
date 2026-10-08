import { Model, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

export interface Dims {
  readonly vocab: number
  readonly seq: number
  readonly d: number
  readonly heads: number
  readonly dMlp: number
}

export const defaultDims = (p: number): Dims => ({ vocab: p + 1, seq: 3, d: 128, heads: 4, dMlp: 512 })

const normal = (fanIn: number) => ({ _tag: "Normal", scale: 1 / Math.sqrt(fanIn) }) as const

export const parameterSpecs = ({ vocab, seq, d, dMlp }: Dims): ReadonlyArray<Model.ParameterSpec> => [
  { name: "embed.W_E", shape: [vocab, d], initializer: normal(d) },
  { name: "embed.W_pos", shape: [seq, d], initializer: normal(d) },
  { name: "attn.W_qkv", shape: [d, 3 * d], initializer: normal(d) },
  { name: "attn.W_o", shape: [d, d], initializer: normal(d) },
  { name: "mlp.W_in", shape: [d, dMlp], initializer: normal(d) },
  { name: "mlp.b_in", shape: [dMlp], initializer: { _tag: "Constant", value: 0 } },
  { name: "mlp.W_out", shape: [dMlp, d], initializer: normal(dMlp) },
  { name: "unembed.W_U", shape: [d, vocab], initializer: normal(d) }
]

const causalMask = (seq: number) =>
  Float32Array.from({ length: seq * seq }, (_, i) => (i % seq > Math.floor(i / seq) ? -1e9 : 0))

/** `[B, seq, d]` -> rows of position `pos`, `[B, d]`. */
const position = (t: Tensor.Any, pos: number) =>
  Effect.gen(function*() {
    const [b, , d] = t.shape
    const row = yield* Tensor.slice(t, { start: [0, pos, 0], end: [b ?? 0, pos + 1, d ?? 0] })
    return yield* Tensor.reshape(row, [b ?? 0, d ?? 0])
  })

/** `[B, S, k] x [k, n]` as one 2-D matmul; batched 3-D x 2-D matmul is much slower on Metal. */
const project = (x: Tensor.Any, w: Tensor.Any) =>
  Effect.gen(function*() {
    const [b = 0, s = 0, k = 0] = x.shape
    const flat = yield* Tensor.matmul(yield* Tensor.reshape(x, [b * s, k]), w)
    return yield* Tensor.reshape(flat, [b, s, w.shape[1] ?? 0])
  })

/**
 * One-layer attention + MLP transformer without LayerNorm (Nanda's grokking setup).
 * Taps: `embed`, `attn.pattern`, `attn.out`, `mlp.hidden`, `resid.final`, `logits`.
 */
export const make = (dims: Dims) =>
  Model.define({
    parameterSpecs: parameterSpecs(dims),
    forward: (params, x) =>
      Effect.gen(function*() {
        const [W_E, W_pos, W_qkv, W_o, W_in, b_in, W_out, W_U] = params
        if (!W_E || !W_pos || !W_qkv || !W_o || !W_in || !b_in || !W_out || !W_U) {
          return yield* new Model.ModelError({ op: "forward", message: `expected 8 parameters, got ${params.length}` })
        }
        const { seq, d, heads } = dims
        const dh = d / heads
        const b = x.shape[0] ?? 0

        const h0 = yield* Tensor.expose(yield* Tensor.add(yield* Tensor.embedding(x, { weight: W_E }), W_pos), "embed")
        const qkv = yield* project(h0, W_qkv)
        const head = (offset: number) =>
          Tensor.slice(qkv, { start: [0, 0, offset], end: [b, seq, offset + d] }).pipe(
            Effect.flatMap(Tensor.reshape([b, seq, heads, dh])),
            Effect.flatMap(Tensor.transpose([0, 2, 1, 3]))
          )
        const q = yield* head(0)
        const k = yield* head(d)
        const v = yield* head(2 * d)

        // q.k as broadcast mul + sum: tiny batched matmuls are slower on Metal
        const qa = yield* Tensor.reshape(q, [b, heads, seq, 1, dh])
        const kb = yield* Tensor.reshape(k, [b, heads, 1, seq, dh])
        const raw = yield* Tensor.sum(yield* Tensor.mul(qa, kb), { dims: [-1] })
        const scaled = yield* Tensor.mul(raw, yield* Tensor.constantLike(raw, 1 / Math.sqrt(dh)))
        const masked = yield* Tensor.add(scaled, yield* Tensor.fromTypedArray(causalMask(seq), [seq, seq]))
        const pattern = yield* Tensor.expose(yield* Tensor.softmax(masked, { dims: [-1] }), "attn.pattern")

        const mixed = yield* Tensor.matmul(pattern, v).pipe(
          Effect.flatMap(Tensor.transpose([0, 2, 1, 3])),
          Effect.flatMap(Tensor.reshape([b, seq, d]))
        )
        const h1 = yield* Tensor.expose(yield* Tensor.add(h0, yield* project(mixed, W_o)), "attn.out")

        const hidden = yield* Tensor.relu(yield* Tensor.add(yield* project(h1, W_in), b_in))
        const hiddenLast = yield* Tensor.expose(yield* position(hidden, seq - 1), "mlp.hidden")
        const h2 = yield* Tensor.expose(
          yield* Tensor.add(yield* position(h1, seq - 1), yield* Tensor.matmul(hiddenLast, W_out)),
          "resid.final"
        )
        return yield* Tensor.expose(yield* Tensor.matmul(h2, W_U), "logits")
      })
  })
