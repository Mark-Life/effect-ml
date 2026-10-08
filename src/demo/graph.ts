import { Gradient, Loss, type Model, type Runtime, Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import type { GraphMessage, GraphSnapshot } from "../protocol.ts"

// Compile-time guard: core's snapshot must stay assignable to the wire schema.
const _snapshotFits: (s: Runtime.GraphSnapshot) => GraphSnapshot = (s) => s
void _snapshotFits

/**
 * Snapshots the forward (`logits`, `loss`) and train (`loss`, grads) graphs over input placeholders.
 * Slots: parameters in spec order, then `input`, then `target`. Nothing executes.
 */
export const snapshots = (
  { model, params, input, target }: {
    model: Model.Definition
    params: ReadonlyArray<Tensor.Any>
    input: Tensor.Any
    target: Tensor.Any
  }
) =>
  Effect.gen(function*() {
    const slotParams = yield* Effect.forEach(params, (p, i) => Tensor.makeInput(i, p))
    const x = yield* Tensor.makeInput(params.length, input)
    const y = yield* Tensor.makeInput(params.length + 1, target)
    const logits = yield* model.forward(slotParams, x)
    const loss = yield* Loss.crossEntropy(logits, y)
    const grads = yield* Gradient.grad(loss, slotParams)
    const slotLabels = [...model.parameterSpecs.map((s) => s.name), "input", "target"]
    const message = (graphId: GraphMessage["graphId"], snapshot: GraphSnapshot): GraphMessage => ({
      _tag: "Graph",
      graphId,
      slotLabels,
      snapshot
    })
    return [
      message("forward", yield* Tensor.graphSnapshot([logits, loss])),
      message("train", yield* Tensor.graphSnapshot([loss, ...grads]))
    ] as const
  })
