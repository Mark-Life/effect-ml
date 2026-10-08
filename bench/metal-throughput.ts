import * as BackendApple from "@effect-torch/backend-apple-native"
import { Gradient, LearningRate, Loss, Model, Optimizer, Tensor, Trainer } from "@effect-torch/core"
import { Duration, Effect } from "effect"

const N = 4096
const ITERS = 20

const benchMatmul = (dtype: "f32" | "bf16") =>
  Effect.gen(function*() {
    const [a, b] = yield* Tensor.compute([yield* Tensor.randn([N, N], { dtype }), yield* Tensor.randn([N, N], { dtype })])
    const mm = yield* Tensor.compile(([x, y]) => Effect.map(Tensor.matmul(x!, y!), (o) => [o]))
    for (let i = 0; i < 3; i++) yield* Tensor.clearAll(yield* mm.call([a!, b!]))
    const t0 = performance.now()
    for (let i = 0; i < ITERS; i++) yield* Tensor.clearAll(yield* mm.call([a!, b!]))
    const ms = (performance.now() - t0) / ITERS
    const tflops = (2 * N ** 3) / (ms / 1000) / 1e12
    console.log(`matmul ${dtype} ${N}^3: ${ms.toFixed(2)} ms/iter, ${tflops.toFixed(2)} TFLOPS`)
  })

const D = 512, LAYERS = 6, HEADS = 8, SEQ = 256, BATCH = 16, VOCAB = 8192

const block = (i: number) =>
  Effect.gen(function*() {
    const attn = yield* Model.residual(yield* Model.chain(
      yield* Model.layerNorm(`b${i}.ln1`, D),
      yield* Model.multiHeadAttention(`b${i}.attn`, D, HEADS, { causal: true, rope: 10000 })
    ))
    const mlp = yield* Model.residual(yield* Model.chain(
      yield* Model.layerNorm(`b${i}.ln2`, D),
      yield* Model.linear(`b${i}.fc1`, D, 4 * D),
      yield* Model.gelu(),
      yield* Model.linear(`b${i}.fc2`, 4 * D, D)
    ))
    return yield* Model.chain(attn, mlp)
  })

const benchTrainer = (precision: "f32" | "mixedBf16", steps: number) =>
  Effect.gen(function*() {
    const blocks = []
    for (let i = 0; i < LAYERS; i++) blocks.push(yield* block(i))
    const model = yield* Model.chain(
      yield* Model.embedding("tok", VOCAB, D),
      ...blocks,
      yield* Model.layerNorm("lnf", D),
      yield* Model.linear("head", D, VOCAB)
    )
    const nParams = model.parameterSpecs.reduce((s, p) => s + p.shape.reduce((a, b) => a * b, 1), 0)
    const toks = new Uint32Array(BATCH * SEQ).map(() => Math.floor(Math.random() * VOCAB))
    const tgts = new Uint32Array(BATCH * SEQ).map(() => Math.floor(Math.random() * VOCAB))
    const [input, target] = yield* Tensor.compute([
      yield* Tensor.fromTypedArray(toks, [BATCH, SEQ]),
      yield* Tensor.fromTypedArray(tgts, [BATCH, SEQ])
    ])
    const times: Array<number> = []
    let last = 0
    const trainer = yield* Trainer.make(model, {
      optimizer: yield* Optimizer.adamW({ weightDecay: 0.01 }),
      lr: LearningRate.constant(3e-4),
      loss: Loss.crossEntropy,
      data: { input: input!, target: target! },
      precision,
      stop: ({ step }) => step >= steps,
      onStep: ({ step, loss, elapsed }) =>
        Effect.sync(() => {
          const ms = Duration.toMillis(elapsed)
          times.push(ms - last)
          last = ms
          if (step === 1 || step === steps) console.log(`  step ${step} loss=${loss.toFixed(4)}`)
        })
    })
    yield* trainer.train(yield* Model.initialize(model))
    const steady = times.slice(3)
    const avg = steady.reduce((a, b) => a + b, 0) / steady.length
    const tokPerS = (BATCH * SEQ) / (avg / 1000)
    // ~6*N*tokens for fwd+bwd, ignoring attention
    const tflops = (6 * nParams * BATCH * SEQ) / (avg / 1000) / 1e12
    console.log(
      `trainer ${precision}: params=${(nParams / 1e6).toFixed(1)}M first-step=${times[0]!.toFixed(0)}ms ` +
        `steady=${avg.toFixed(1)}ms/step (min ${Math.min(...steady).toFixed(1)}) ${tokPerS.toFixed(0)} tok/s ~${tflops.toFixed(2)} TFLOPS(6ND)`
    )
  })

const probe = (name: string, eff: Effect.Effect<unknown, unknown, any>) =>
  eff.pipe(
    Effect.tap(() => Effect.sync(() => console.log(`probe ${name}: ok`))),
    Effect.catchCause((c) => Effect.sync(() => console.log(`probe ${name}: FAIL ${String(c).split("\n")[0]!.slice(0, 200)}`)))
  )

const probes = Effect.gen(function*() {
  const x = yield* Tensor.randn([4, 8])
  const w = yield* Tensor.ones([8])
  yield* probe("rmsNorm grad", Effect.gen(function*() {
    const y = yield* Tensor.sum(yield* Tensor.rmsNorm(x, w))
    yield* Tensor.compute(yield* Gradient.grad(y, [w]))
  }))
  yield* probe("jvp tanh", Effect.gen(function*() {
    const y = yield* Tensor.tanh(x)
    yield* Tensor.compute([yield* Gradient.jvp(y, x, yield* Tensor.onesLike(x))])
  }))
  yield* probe("jvp layerNorm", Effect.gen(function*() {
    const y = yield* Tensor.layerNorm(x, w, yield* Tensor.zeros([8]))
    yield* Tensor.compute([yield* Gradient.jvp(y, x, yield* Tensor.onesLike(x))])
  }))
  const q = yield* Tensor.randn([2, 4, 16, 32])
  yield* probe("sdpa grad", Effect.gen(function*() {
    const y = yield* Tensor.sum(yield* Tensor.scaledDotProductAttention(q, q, q, { causal: true }))
    yield* Tensor.compute(yield* Gradient.grad(y, [q]))
  }))
  yield* probe("jvp sdpa", Effect.gen(function*() {
    const y = yield* Tensor.scaledDotProductAttention(q, q, q, { causal: true })
    yield* Tensor.compute([yield* Gradient.jvp(y, q, yield* Tensor.onesLike(q))])
  }))
  yield* probe("vmap matmul", Effect.gen(function*() {
    const xb = yield* Tensor.randn([5, 4, 8])
    const y = yield* Tensor.matmul(x, yield* Tensor.randn([8, 3]))
    yield* Tensor.compute([yield* Gradient.vmap(y, x, xb)])
  }))
  const img = yield* Tensor.randn([2, 3, 16, 16])
  const k = yield* Tensor.randn([8, 3, 3, 3])
  yield* probe("conv2d grad", Effect.gen(function*() {
    const y = yield* Tensor.sum(yield* Tensor.conv2d(img, k, { padding: 1 } as any))
    yield* Tensor.compute(yield* Gradient.grad(y, [img, k]))
  }))
  yield* probe("convTranspose2d grad", Effect.gen(function*() {
    const kt = yield* Tensor.randn([3, 8, 4, 4])
    const y = yield* Tensor.sum(yield* Tensor.convTranspose2d(img, kt, { stride: 2, padding: 1 } as any))
    yield* Tensor.compute(yield* Gradient.grad(y, [img, kt]))
  }))
  yield* probe("bf16 conv2d", Effect.gen(function*() {
    const y = yield* Tensor.conv2d(yield* Tensor.cast(img, "bf16" as any), yield* Tensor.cast(k, "bf16" as any), { padding: 1 } as any)
    yield* Tensor.compute([y])
  }))
  yield* probe("gather+scatterAdd grad", Effect.gen(function*() {
    const idx = yield* Tensor.fromTypedArray(new Uint32Array([0, 2, 1, 3]), [4, 1])
    const y = yield* Tensor.sum(yield* Tensor.gather(x, idx, { dim: 1 }))
    yield* Tensor.compute(yield* Gradient.grad(y, [x]))
  }))
  yield* probe("cumsum grad", Effect.gen(function*() {
    const y = yield* Tensor.sum(yield* Tensor.cumsum(x, 1))
    yield* Tensor.compute(yield* Gradient.grad(y, [x]))
  }))
})

const which = process.argv[2] ?? "all"
const program = Effect.gen(function*() {
  if (which === "all" || which === "probe") yield* probes
  if (which === "all" || which === "mm") {
    yield* benchMatmul("f32")
    yield* benchMatmul("bf16")
  }
  if (which === "all" || which === "train") {
    yield* benchTrainer("f32", 15)
    yield* benchTrainer("mixedBf16", 15)
  }
})

await Effect.runPromise(program.pipe(Effect.provide(BackendApple.layer())) as Effect.Effect<void>)
