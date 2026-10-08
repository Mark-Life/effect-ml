import * as BackendApple from "@effect-torch/backend-apple-native"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const run = (n: number, dtype: "f32" | "bf16" | "f16", chain: number) =>
  Effect.gen(function*() {
    const [a, b] = yield* Tensor.compute([yield* Tensor.randn([n, n], { dtype }), yield* Tensor.randn([n, n], { dtype })])
    const mm = yield* Tensor.compile(([x, y]) =>
      Effect.gen(function*() {
        let z: Tensor.Any = x!
        for (let i = 0; i < chain; i++) z = yield* Tensor.matmul(z, y!)
        return [z]
      }))
    for (let i = 0; i < 2; i++) yield* Tensor.clearAll(yield* mm.call([a!, b!]))
    const iters = 5
    const t0 = performance.now()
    for (let i = 0; i < iters; i++) yield* Tensor.clearAll(yield* mm.call([a!, b!]))
    const ms = (performance.now() - t0) / iters
    console.log(`n=${n} ${dtype} chain=${chain}: ${(ms / chain).toFixed(2)} ms/matmul ${((2 * n ** 3 * chain) / (ms / 1000) / 1e12).toFixed(2)} TFLOPS`)
  })

await Effect.runPromise(Effect.gen(function*() {
  yield* run(4096, "f32", 8)
  yield* run(4096, "bf16", 8)
  yield* run(4096, "f16", 8)
  yield* run(2048, "f32", 8)
  yield* run(2048, "bf16", 8)
}).pipe(Effect.provide(BackendApple.layer())) as Effect.Effect<void>)
