import { Cause, Effect, Exit, Fiber } from "effect"

/** Runs `effect` as the process main: Ctrl-C interrupts it and exits 0, failures print their cause and exit 1. */
export const runMain = <A, E>(effect: Effect.Effect<A, E>) => {
  const fiber = Effect.runFork(effect)
  const stop = () => void Effect.runFork(Fiber.interrupt(fiber))
  process.once("SIGINT", stop)
  process.once("SIGTERM", stop)
  fiber.addObserver((exit) => {
    const failed = Exit.isFailure(exit) && !Cause.hasInterruptsOnly(exit.cause)
    if (failed) console.error(Cause.pretty(exit.cause))
    process.exit(failed ? 1 : 0)
  })
}
