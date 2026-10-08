import { LearningRate, Loss, Model, Optimizer, Runtime, Tensor, Trainer } from "@effect-torch/core"
import { Duration, Effect } from "effect"
import { makeTensors, modularAddition } from "./demo/data.ts"
import { snapshots } from "./demo/graph.ts"
import * as DemoModel from "./demo/model.ts"
import * as ProbeFn from "./demo/probe-fn.ts"
import { evalPoint, probeFrames, type Read } from "./demo/probe-views.ts"
import type { DiagnosticsMessage, MetricPoint } from "./protocol.ts"
import { ViewerServer } from "./server.ts"

export interface DemoConfig {
  readonly p: number
  readonly trainFrac: number
  readonly totalSteps: number
  /** 0 disables probes: one uninterrupted `train` call. */
  readonly probeEvery: number
  readonly lr: number
  readonly weightDecay: number
}

export interface RunSummary {
  readonly steps: number
  readonly wallMs: number
  readonly stepMs: { readonly median: number; readonly mean: number; readonly p90: number }
  readonly probes: number
  readonly probeMsTotal: number
  readonly probeMsMean: number
  readonly finalTrainLoss: number
  readonly finalEval?: { readonly trainAcc: number; readonly testAcc: number; readonly testLoss: number }
}

const toF32 = (op: string, t: Tensor.Any) =>
  Tensor.toTypedArray(t).pipe(
    Effect.flatMap((data) =>
      data instanceof Float32Array
        ? Effect.succeed<Read>({ data, shape: t.shape })
        : Effect.fail(new Tensor.TensorError({ op, message: `expected f32, got ${t.dtype}` }))
    )
  )

const quantile = (sorted: ReadonlyArray<number>, q: number) =>
  sorted.length === 0 ? 0 : sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]!

const diagnosticsMessage = (
  programId: DiagnosticsMessage["programId"],
  entries: ReadonlyArray<Tensor.ProgramDiagnostics>
): ReadonlyArray<DiagnosticsMessage> => {
  const last = entries[entries.length - 1]
  return last ? [{ _tag: "Diagnostics", programId, signature: last.signature, diagnostics: last.diagnostics }] : []
}

export const run = Effect.fn("TrainLoop.run")(function*(config: DemoConfig) {
  const runtime = yield* Runtime.Runtime
  const viewer = yield* ViewerServer
  const started = performance.now()

  const split = modularAddition({ p: config.p, trainFrac: config.trainFrac })
  const data = yield* makeTensors(split)
  const dims = DemoModel.defaultDims(config.p)
  const model = yield* DemoModel.make(dims)
  const paramNames = model.parameterSpecs.map((s) => s.name)

  yield* viewer.publish({
    _tag: "RunInfo",
    runId: Date.now().toString(36),
    task: `modular-addition p=${config.p}, train ${split.trainIndices.length}/${split.n}`,
    backend: runtime.backend.name,
    totalSteps: config.totalSteps,
    probeEvery: config.probeEvery,
    parameters: model.parameterSpecs.map(({ name, shape }) => ({ name, shape }))
  })
  yield* viewer.publish({ _tag: "Status", state: "compiling" })

  const initial = yield* Tensor.compute(yield* Model.initialize(model))
  const graphs = yield* snapshots({ model, params: initial, input: data.trainX, target: data.trainY })
  yield* Effect.forEach(graphs, viewer.publish, { discard: true })
  yield* Effect.logInfo(`graph nodes: forward ${graphs[0].snapshot.nodes.length}, train ${graphs[1].snapshot.nodes.length}`)

  const optimizer = yield* Optimizer.adamW({ beta1: 0.9, beta2: 0.98, weightDecay: config.weightDecay })
  const stepTimes: Array<number> = []
  let chunkEnd = 0
  let lastElapsed = 0
  let lastLoss = Number.NaN
  let skipped = 0

  const onStep = ({ step, loss, elapsed }: Trainer.TrainStep) =>
    Effect.gen(function*() {
      const ms = Duration.toMillis(elapsed)
      const stepMs = ms - lastElapsed
      lastElapsed = ms
      lastLoss = loss
      stepTimes.push(stepMs)
      if (!Number.isFinite(loss)) {
        skipped++
        return
      }
      const memoryBytes = yield* runtime.extensions.diagnostics.externalMemoryBytes
      const point: MetricPoint = { step, loss, stepMs, memoryBytes }
      yield* viewer.pushMetric(point)
    })

  const trainer = yield* Trainer.make(model, {
    optimizer,
    lr: LearningRate.withWarmup(LearningRate.constant(config.lr), 10),
    loss: (pred, target) => Loss.crossEntropy(pred, target),
    data: { input: data.trainX, target: data.trainY },
    stop: ({ step }) => step >= chunkEnd,
    onStep
  })
  const probeFn = yield* ProbeFn.make(model)

  const probe = Effect.fn("TrainLoop.probe")(function*(step: number, params: ReadonlyArray<Tensor.Concrete>) {
    const t0 = performance.now()
    const outputs = yield* probeFn.call([...params, data.allX, data.allY, data.trainMask])
    const reads = yield* Effect.forEach(outputs, (t) => toF32("probe", t)).pipe(
      Effect.ensuring(Tensor.clearAll(outputs))
    )
    const paramReads = yield* Effect.forEach(params, (t) => toF32("probe", t))
    const [perExample, correct, pattern, hidden, ...grads] = reads
    if (!perExample || !correct || !pattern || !hidden) {
      return yield* new Tensor.TensorError({ op: "probe", message: `probe returned ${reads.length} outputs` })
    }
    const read = {
      step,
      p: config.p,
      heads: dims.heads,
      seq: dims.seq,
      paramNames,
      params: paramReads,
      grads,
      perExample: perExample.data,
      correct: correct.data,
      trainMask: split.trainMask,
      pattern,
      hidden
    }
    const frames = probeFrames(read)
    const probeMs = performance.now() - t0
    const point = evalPoint(read, probeMs)
    const finite = [point.trainLoss, point.testLoss, point.trainAcc, point.testAcc].every(Number.isFinite)
    if (finite) yield* viewer.publish(point)
    else yield* Effect.logWarning(`step ${step}: non-finite eval, skipped`)
    yield* Effect.forEach(frames, ({ header, payload }) => viewer.publishProbe(header, payload), { discard: true })
    return point
  })

  const chunk = config.probeEvery > 0 ? config.probeEvery : config.totalSteps
  let params: ReadonlyArray<Tensor.Concrete> = initial
  let resume: Trainer.Resume<Optimizer.AdamState> | undefined
  let probes = 0
  let probeMsTotal = 0
  let lastEval: ReturnType<typeof evalPoint> | undefined

  yield* viewer.publish({ _tag: "Status", state: "training" })
  while ((resume?.step ?? 0) < config.totalSteps) {
    chunkEnd = Math.min(config.totalSteps, (resume?.step ?? 0) + chunk)
    lastElapsed = 0
    const trained = yield* trainer.train(params, resume)
    const stale = resume ? optimizer.stateRoots(resume.state).filter(Tensor.isTensor) : []
    yield* Tensor.clearAll([...params, ...stale])
    params = trained.params
    resume = { state: trained.state, step: trained.step }

    if (probes === 0) {
      const messages = diagnosticsMessage("train-step", yield* trainer.diagnostics)
      yield* Effect.forEach(messages, viewer.publish, { discard: true })
    }
    if (config.probeEvery > 0) {
      lastEval = yield* probe(trained.step, params)
      probeMsTotal += lastEval.probeMs
      if (probes === 0) {
        const messages = diagnosticsMessage("probe", yield* probeFn.diagnostics)
        yield* Effect.forEach(messages, viewer.publish, { discard: true })
      }
      probes++
    }
  }

  const sorted = stepTimes.slice(1).sort((a, b) => a - b)
  const summary: RunSummary = {
    steps: resume?.step ?? 0,
    wallMs: performance.now() - started,
    stepMs: {
      median: quantile(sorted, 0.5),
      mean: sorted.reduce((a, b) => a + b, 0) / Math.max(1, sorted.length),
      p90: quantile(sorted, 0.9)
    },
    probes,
    probeMsTotal,
    probeMsMean: probes === 0 ? 0 : probeMsTotal / probes,
    finalTrainLoss: lastLoss,
    ...(lastEval
      ? { finalEval: { trainAcc: lastEval.trainAcc, testAcc: lastEval.testAcc, testLoss: lastEval.testLoss } }
      : {})
  }
  if (skipped > 0) yield* Effect.logWarning(`${skipped} steps had non-finite loss and were not charted`)
  yield* viewer.publish({ _tag: "Status", state: "done", message: `${summary.steps} steps` })
  const stale = resume ? optimizer.stateRoots(resume.state).filter(Tensor.isTensor) : []
  yield* Tensor.clearAll([...params, ...stale])
  return summary
})
