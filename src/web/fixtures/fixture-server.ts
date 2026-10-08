import { Config, Effect, Schedule, Schema } from "effect"
import { GraphSnapshot, type MetricPoint } from "../../protocol.ts"
import { runMain } from "../../run-main.ts"
import * as Viewer from "../../server.ts"
import { evalPoint, makeProbeSource, mulberry32, parameters, P, stepLoss, syntheticDiagnostics } from "./synthetic.ts"

const Fixture = Schema.Struct({
  slotLabels: Schema.Array(Schema.String),
  forward: GraphSnapshot,
  train: GraphSnapshot
})

const program = Effect.gen(function*() {
  const viewer = yield* Viewer.ViewerServer
  const totalSteps = yield* Config.int("FIXTURE_STEPS").pipe(Config.withDefault(20_000))
  const probeEvery = yield* Config.int("FIXTURE_PROBE_EVERY").pipe(Config.withDefault(100))
  const stepsPerTick = yield* Config.int("FIXTURE_STEPS_PER_TICK").pipe(Config.withDefault(4))
  const fixture = yield* Effect.promise(() => Bun.file(new URL("./graph.json", import.meta.url)).json()).pipe(
    Effect.flatMap(Schema.decodeUnknownEffect(Fixture))
  )
  const grokAt = Math.round(totalSteps * 0.4)
  const rng = mulberry32(1)
  const probes = makeProbeSource(2)

  yield* viewer.publish({
    _tag: "RunInfo",
    runId: Date.now().toString(36),
    task: `modular-addition p=${P} (fixture)`,
    backend: "fixture",
    totalSteps,
    probeEvery,
    parameters
  })
  yield* viewer.publish({ _tag: "Status", state: "compiling" })
  yield* viewer.publish({ _tag: "Graph", graphId: "forward", slotLabels: fixture.slotLabels, snapshot: fixture.forward })
  yield* viewer.publish({ _tag: "Graph", graphId: "train", slotLabels: fixture.slotLabels, snapshot: fixture.train })
  yield* viewer.publish(syntheticDiagnostics("train-step"))
  yield* viewer.publish(syntheticDiagnostics("probe"))
  yield* viewer.publish({ _tag: "Status", state: "training" })

  let step = 0
  let memory = 182_000_000
  const tick = Effect.gen(function*() {
    for (let i = 0; i < stepsPerTick && step < totalSteps; i++) {
      step++
      memory += Math.round((rng() - 0.5) * 400_000)
      const point: MetricPoint = { step, loss: stepLoss(step, rng), stepMs: 3.8 + rng() * 0.8, memoryBytes: memory }
      yield* viewer.pushMetric(point)
      if (step % probeEvery !== 0) continue
      const started = performance.now()
      const accuracy = evalPoint({ step, grokAt, rng, probeMs: 0 })
      const frames = probes(step, grokAt, accuracy)
      yield* viewer.publish({ ...accuracy, probeMs: performance.now() - started })
      yield* Effect.forEach(frames, ({ header, payload }) => viewer.publishProbe(header, payload), { discard: true })
    }
  })

  yield* tick.pipe(Effect.repeat({ schedule: Schedule.spaced("20 millis"), until: () => step >= totalSteps }))
  yield* viewer.publish({ _tag: "Status", state: "done", message: `${totalSteps} steps` })
  return yield* Effect.never
})

runMain(program.pipe(Effect.provide(Viewer.layer)))

export {}
