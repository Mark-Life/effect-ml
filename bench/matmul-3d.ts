import * as BackendApple from "@effect-torch/backend-apple-native"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const time = (label: string, build: (i: ReadonlyArray<Tensor.Lazy>) => Effect.Effect<Tensor.Lazy, any, any>, inputs: ReadonlyArray<Tensor.Concrete>) => Effect.gen(function*() {
  const fn = yield* Tensor.compile((i) => Effect.map(build(i), (o) => [o]))
  for (let i = 0; i < 2; i++) { const o = yield* fn.call(inputs); yield* Tensor.clearAll(o) }
  const t0 = performance.now()
  for (let i = 0; i < 10; i++) { const o = yield* fn.call(inputs); yield* Tensor.toTypedArray(o[0]!); yield* Tensor.clearAll(o) }
  console.log(label, ((performance.now() - t0) / 10).toFixed(2), "ms")
})

const program = Effect.gen(function*() {
  const [x, w, x2] = yield* Tensor.compute([yield* Tensor.randn([3831, 3, 128]), yield* Tensor.randn([128, 384]), yield* Tensor.randn([11493, 128])] as const)
  const [q, kt] = yield* Tensor.compute([yield* Tensor.randn([3831, 4, 3, 32]), yield* Tensor.randn([3831, 4, 32, 3])] as const)
  yield* time("attn 4d mm", ([a, b]) => Tensor.matmul(a!, b!), [q, kt])
  yield* time("attn mul+sum", ([a, b]) => Effect.gen(function*() {
    const qa = yield* Tensor.reshape(a!, [3831, 4, 3, 1, 32])
    const kb = yield* Tensor.reshape(yield* Tensor.transpose(b!, [0, 1, 3, 2]), [3831, 4, 1, 3, 32])
    return yield* Tensor.sum(yield* Tensor.mul(qa, kb), { dims: [-1] })
  }), [q, kt])
  yield* time("3d x 2d", ([a, b]) => Effect.flatMap(Tensor.matmul(a!, b!), (m) => Tensor.sum(m)), [x, w])
  yield* time("reshape 2d x 2d", ([a, b]) => Effect.gen(function*() { return yield* Tensor.sum(yield* Tensor.matmul(yield* Tensor.reshape(a!, [11493, 128]), b!)) }), [x, w])
  yield* time("2d x 2d", ([a, b]) => Effect.flatMap(Tensor.matmul(a!, b!), (m) => Tensor.sum(m)), [x2, w])
  yield* time("sum only", ([a]) => Tensor.sum(a!), [x2])
  yield* time("rowsum", ([a]) => Tensor.sum(a!, { dims: [-1] }), [x2])
  yield* time("exp", ([a]) => Tensor.exp(a!), [x2])
  yield* time("colsum", ([a]) => Tensor.sum(a!, { dims: [0] }), [x2])
  yield* time("3d mm no sum", ([a, b]) => Tensor.matmul(a!, b!), [x, w])
  yield* time("tiny", ([a]) => Tensor.exp(a!), [w])
  yield* time("2d mm no sum", ([a, b]) => Tensor.matmul(a!, b!), [x2, w])
})
await Effect.runPromise(program.pipe(Effect.provide(BackendApple.layer())) as Effect.Effect<void>)
