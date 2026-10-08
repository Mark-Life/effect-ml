import * as BackendApple from "@effect-torch/backend-apple-native"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const program = Effect.gen(function*() {
  const a = yield* Tensor.randn([512, 512])
  const b = yield* Tensor.randn([512, 512])
  const mean = yield* Tensor.mean(yield* Tensor.matmul(a, b))
  const [value] = yield* Tensor.compute([mean])
  return yield* Tensor.toNumberArray(value)
})

const result = await Effect.runPromise(program.pipe(Effect.provide(BackendApple.layer())))
console.log("mean:", result[0])
