import { Gradient, Loss, type Model, Runtime, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

/** Taps returned by the probe program, in output order. */
export const probeTaps = ["attn.pattern", "mlp.hidden"] as const

/**
 * Compiled probe over the full `p*p` grid.
 * Inputs: `[...params, allX, allY, trainMask]`.
 * Outputs: `[perExampleLoss, correct, ...probeTaps, ...grads]` where grads are of the mean train-subset loss.
 */
export const make = (model: Model.Definition) =>
  Tensor.compile((inputs) =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      const count = model.parameterSpecs.length
      const params = inputs.slice(0, count)
      const [allX, allY, trainMask] = inputs.slice(count)
      if (!allX || !allY || !trainMask) {
        return yield* new Tensor.TensorError({ op: "probe", message: `expected ${count + 3} inputs` })
      }

      const logits = yield* model.forward(params, allX)
      const perExample = yield* Loss.crossEntropy(logits, allY, { reduction: "none" })
      const correct = yield* Tensor.argmax(logits, -1).pipe(
        Effect.flatMap((predicted) => Tensor.eq(predicted, allY)),
        Effect.flatMap(Tensor.cast("f32"))
      )
      const trainLoss = yield* Tensor.div(
        yield* Tensor.sum(yield* Tensor.mul(perExample, trainMask)),
        yield* Tensor.sum(trainMask)
      )
      const grads = yield* Gradient.grad(trainLoss, params)

      const exposures = yield* runtime.exposures(logits).pipe(
        Effect.mapError((backend) => new Tensor.TensorError({ op: "probe", message: backend.message, backend }))
      )
      const taps = probeTaps.map((name) => exposures.find((e) => e.name === name)?.tensor)
      const missing = probeTaps.filter((_, i) => taps[i] === undefined)
      if (missing.length > 0) {
        return yield* new Tensor.TensorError({ op: "probe", message: `missing taps: ${missing.join(", ")}` })
      }
      return [perExample, correct, ...taps.filter((t) => t !== undefined), ...grads]
    })
  )

export type ProbeFn = Effect.Success<ReturnType<typeof make>>
