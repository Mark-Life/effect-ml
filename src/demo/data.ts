import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import { mulberry32 } from "../probes.ts"

export interface SplitOptions {
  readonly p: number
  readonly trainFrac: number
  readonly seed?: number
}

/** All `p*p` pairs `[a, b, =]` in `a*p + b` order, plus a seeded train subset (sorted indices). */
export const modularAddition = ({ p, trainFrac, seed = 0 }: SplitOptions) => {
  const n = p * p
  const rng = mulberry32(seed)
  const order = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const t = order[i]!
    order[i] = order[j]!
    order[j] = t
  }
  const trainIndices = order.slice(0, Math.round(n * trainFrac)).sort((x, y) => x - y)
  const trainMask = new Float32Array(n)
  for (const i of trainIndices) trainMask[i] = 1
  return { p, n, equalsToken: p, vocab: p + 1, trainIndices, trainMask }
}

export type Split = ReturnType<typeof modularAddition>

const rowsFor = (indices: ReadonlyArray<number>, p: number) => {
  const x = new BigInt64Array(indices.length * 3)
  const y = new BigInt64Array(indices.length)
  indices.forEach((ex, row) => {
    const a = Math.floor(ex / p)
    const b = ex % p
    x[row * 3] = BigInt(a)
    x[row * 3 + 1] = BigInt(b)
    x[row * 3 + 2] = BigInt(p)
    y[row] = BigInt((a + b) % p)
  })
  return { x, y }
}

/** Materialized device tensors: train subset for the Trainer, full grid for probes. */
export const makeTensors = (split: Split) =>
  Effect.gen(function*() {
    const train = rowsFor(split.trainIndices, split.p)
    const all = rowsFor(Array.from({ length: split.n }, (_, i) => i), split.p)
    const [trainX, trainY, allX, allY, trainMask] = yield* Tensor.compute([
      yield* Tensor.fromTypedArray(train.x, [split.trainIndices.length, 3]),
      yield* Tensor.fromTypedArray(train.y, [split.trainIndices.length]),
      yield* Tensor.fromTypedArray(all.x, [split.n, 3]),
      yield* Tensor.fromTypedArray(all.y, [split.n]),
      yield* Tensor.fromTypedArray(split.trainMask, [split.n])
    ] as const)
    return { trainX, trainY, allX, allY, trainMask }
  })
