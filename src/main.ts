import * as BackendApple from "@effect-torch/backend-apple-native"
import { Cause, Config, Effect, Exit, Layer } from "effect"
import { runMain } from "./run-main.ts"
import * as Viewer from "./server.ts"
import { run } from "./train-loop.ts"

const config = Config.all({
  p: Config.int("DEMO_P").pipe(Config.withDefault(113)),
  trainFrac: Config.number("DEMO_TRAIN_FRAC").pipe(Config.withDefault(0.3)),
  totalSteps: Config.int("DEMO_STEPS").pipe(Config.withDefault(20_000)),
  probeEvery: Config.int("DEMO_PROBE_EVERY").pipe(Config.withDefault(100)),
  lr: Config.number("DEMO_LR").pipe(Config.withDefault(1e-3)),
  weightDecay: Config.number("DEMO_WEIGHT_DECAY").pipe(Config.withDefault(1))
})

const program = Effect.gen(function*() {
  const viewer = yield* Viewer.ViewerServer
  const hold = yield* Config.boolean("DEMO_HOLD").pipe(Config.withDefault(true))
  const exit = yield* Effect.exit(Effect.gen(function*() {
    return yield* run(yield* config)
  }))
  if (Exit.isSuccess(exit)) {
    yield* Effect.logInfo(`summary ${JSON.stringify(exit.value)}`)
  } else if (!Cause.hasInterruptsOnly(exit.cause)) {
    const message = Cause.pretty(exit.cause)
    yield* viewer.publish({ _tag: "Status", state: "failed", message })
    yield* Effect.logError(message)
  }
  if (hold) return yield* Effect.never
  return yield* exit
})

runMain(program.pipe(Effect.provide(Layer.merge(Viewer.layer, BackendApple.layer()))))
