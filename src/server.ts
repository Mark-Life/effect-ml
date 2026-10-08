import type { ServerWebSocket } from "bun"
import { Config, Context, Effect, Layer, Schedule, Schema } from "effect"
import {
  encodeProbe,
  type DiagnosticsMessage,
  type EvalPoint,
  type GraphMessage,
  type MetricPoint,
  type ProbeHeader,
  type RunInfo,
  type ServerMessage,
  type Status
} from "./protocol.ts"
import index from "./web/index.html"

const TOPIC = "viewer"
const METRICS_CAP = 50_000
/**
 * A socket that buffers past this limit is closed instead of silently losing
 * frames; the client reconnects and receives the full replay. The limit must
 * stay well above the replay size.
 */
const BACKPRESSURE_LIMIT = 16 * 1024 * 1024
/** Probe frames are latest-value snapshots, so slow sockets skip them first. */
const PROBE_SKIP_BYTES = 2 * 1024 * 1024
const CLOSE_GRACE = "200 millis"

export class ViewerServerError extends Schema.TaggedErrorClass<ViewerServerError>()("ViewerServerError", {
  port: Schema.Finite,
  message: Schema.String
}) {}

type Socket = ServerWebSocket<undefined>

interface Replay {
  runInfo?: RunInfo
  status?: Status
  readonly graphs: Map<GraphMessage["graphId"], GraphMessage>
  readonly diagnostics: Map<DiagnosticsMessage["programId"], DiagnosticsMessage>
  metrics: Array<MetricPoint>
  readonly evals: Array<EvalPoint>
  readonly probes: Map<string, Uint8Array>
}

const emptyReplay = (): Replay => ({
  graphs: new Map(),
  diagnostics: new Map(),
  metrics: [],
  evals: [],
  probes: new Map()
})

/** Keeps every point until the cap, then halves resolution by dropping every other point. */
const appendMetrics = (current: Array<MetricPoint>, points: ReadonlyArray<MetricPoint>) => {
  const next = current.concat(points)
  return next.length > METRICS_CAP ? next.filter((_, i) => i % 2 === 0 || i === next.length - 1) : next
}

const record = (replay: Replay, message: ServerMessage) => {
  switch (message._tag) {
    case "RunInfo":
      replay.runInfo = message
      return
    case "Status":
      replay.status = message
      return
    case "Graph":
      replay.graphs.set(message.graphId, message)
      return
    case "Diagnostics":
      replay.diagnostics.set(message.programId, message)
      return
    case "Metrics":
      replay.metrics = appendMetrics(replay.metrics, message.points)
      return
    case "Eval":
      replay.evals.push(message)
      return
  }
}

const replayFrames = (replay: Replay) => {
  const messages: Array<ServerMessage> = [
    ...(replay.runInfo ? [replay.runInfo] : []),
    ...replay.graphs.values(),
    ...replay.diagnostics.values(),
    ...(replay.metrics.length > 0 ? [{ _tag: "Metrics" as const, points: replay.metrics }] : []),
    ...replay.evals,
    ...(replay.status ? [replay.status] : [])
  ]
  return [...messages.map((m) => JSON.stringify(m)), ...replay.probes.values()]
}

export interface Interface {
  readonly url: string
  readonly publish: (message: ServerMessage) => Effect.Effect<void>
  readonly publishProbe: (header: ProbeHeader, payload: Float32Array | Uint32Array) => Effect.Effect<void>
  /** Buffers one point; buffered points go out as a single `Metrics` message every 100 ms. */
  readonly pushMetric: (point: MetricPoint) => Effect.Effect<void>
}

export class ViewerServer extends Context.Service<ViewerServer, Interface>()("effect-ml/ViewerServer") {}

const serve = (port: number, development: boolean, replay: Replay, clients: Set<Socket>) =>
  Effect.acquireRelease(
    Effect.try({
      try: () =>
        Bun.serve({
          port,
          development,
          routes: { "/": index },
          fetch: (request, server) =>
            new URL(request.url).pathname === "/ws" && server.upgrade(request, { data: undefined })
              ? undefined
              : new Response("not found", { status: 404 }),
          websocket: {
            backpressureLimit: BACKPRESSURE_LIMIT,
            closeOnBackpressureLimit: true,
            open: (ws: Socket) => {
              for (const frame of replayFrames(replay)) ws.send(frame)
              ws.subscribe(TOPIC)
              clients.add(ws)
            },
            message: () => {},
            close: (ws: Socket) => {
              clients.delete(ws)
              ws.unsubscribe(TOPIC)
            }
          }
        }),
      catch: (cause) => new ViewerServerError({ port, message: String(cause) })
    }),
    (server) =>
      Effect.gen(function*() {
        // graceful close lets queued frames, such as a final status, reach the browser
        for (const ws of clients) ws.close(1001, "server stopping")
        yield* Effect.sleep(CLOSE_GRACE)
        yield* Effect.promise(() => server.stop(true))
      })
  )

export const layer = Layer.effect(
  ViewerServer,
  Effect.gen(function*() {
    const port = yield* Config.port("VIEWER_PORT").pipe(Config.withDefault(4173))
    const development = yield* Config.boolean("VIEWER_DEV").pipe(Config.withDefault(false))
    const replay = emptyReplay()
    const clients = new Set<Socket>()
    const server = yield* serve(port, development, replay, clients)

    const publish = Effect.fn("ViewerServer.publish")(function*(message: ServerMessage) {
      record(replay, message)
      server.publish(TOPIC, JSON.stringify(message))
    })

    const publishProbe = Effect.fn("ViewerServer.publishProbe")(
      function*(header: ProbeHeader, payload: Float32Array | Uint32Array) {
        const frame = encodeProbe(header, payload)
        replay.probes.set(header.name, frame)
        for (const ws of clients) if (ws.getBufferedAmount() < PROBE_SKIP_BYTES) ws.send(frame)
      }
    )

    let buffer: Array<MetricPoint> = []
    const flush = Effect.suspend(() => {
      if (buffer.length === 0) return Effect.void
      const points = buffer
      buffer = []
      return publish({ _tag: "Metrics", points })
    })
    yield* flush.pipe(Effect.repeat(Schedule.spaced("100 millis")), Effect.forkScoped)
    yield* Effect.addFinalizer(() => flush)

    yield* Effect.logInfo(`viewer on ${server.url.href}`)
    return ViewerServer.of({
      url: server.url.href,
      publish,
      publishProbe,
      pushMetric: (point) => Effect.sync(() => void buffer.push(point))
    })
  })
)
