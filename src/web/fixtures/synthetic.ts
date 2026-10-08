import { fourierNorms, mulberry32, gridChannel, heatmapProbe, histogramProbe, type Matrix, toMatrix } from "../../probes.ts"
import type { DiagnosticsMessage, EvalPoint, RunInfo } from "../../protocol.ts"

export const P = 113
const VOCAB = P + 1
const D = 128
const KEY_FREQS = [14, 35, 41, 42, 52]

export const parameters: RunInfo["parameters"] = [
  { name: "embed.W_E", shape: [VOCAB, D] },
  { name: "embed.W_pos", shape: [3, D] },
  { name: "mlp.W_in", shape: [D, 512] },
  { name: "mlp.b_in", shape: [512] },
  { name: "mlp.W_out", shape: [512, D] },
  { name: "unembed.W_U", shape: [D, VOCAB] }
]

export { mulberry32 }

export type Rng = ReturnType<typeof mulberry32>

const gaussian = (rng: Rng) => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng())

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

/** Generalization progress in [0, 1]: rises sharply around `grokAt`. */
export const grokProgress = (step: number, grokAt: number) => sigmoid((step - grokAt) / (grokAt * 0.08))

export const stepLoss = (step: number, rng: Rng) => 4.8 * Math.exp(-step / 350) + 0.003 + 0.002 * Math.abs(gaussian(rng))

export const evalPoint = (
  { step, grokAt, rng, probeMs }: { step: number; grokAt: number; rng: Rng; probeMs: number }
): EvalPoint => {
  const g = grokProgress(step, grokAt)
  const memorized = 1 - Math.exp(-step / 300)
  return {
    _tag: "Eval",
    step,
    trainLoss: 4.8 * Math.exp(-step / 350) + 0.003,
    testLoss: (1 - g) * (4.7 + 3 * memorized) + g * 0.01 + 0.02 * Math.abs(gaussian(rng)),
    trainAcc: Math.min(1, 0.01 + memorized),
    testAcc: Math.min(1, 0.01 + 0.04 * memorized + 0.95 * g),
    probeMs
  }
}

export const syntheticDiagnostics = (programId: DiagnosticsMessage["programId"]): DiagnosticsMessage => {
  const scale = programId === "probe" ? 0.6 : 1
  const count = (n: number) => Math.round(n * scale)
  return {
    _tag: "Diagnostics",
    programId,
    signature: `fixture|${programId}|f32[114,128],f32[3,128],f32[128,512],f32[512],f32[512,128],f32[128,114],i64[3831,3]`,
    diagnostics: {
      semanticNodesBeforeOptimization: count(412),
      semanticNodesAfterOptimization: count(412),
      instructions: [
        { kind: "fused_elementwise", count: count(58) },
        { kind: "matmul", count: count(21) },
        { kind: "reduce", count: count(14) },
        { kind: "copy", count: count(9) },
        { kind: "gather", count: count(3) },
        { kind: "scatter_add", count: count(2) },
        { kind: "softmax", count: count(2) },
        { kind: "cross_entropy", count: 2 },
        ...(programId === "train-step" ? [{ kind: "adam_w_step", count: 8 }] : [])
      ],
      pipelineCount: count(31),
      commandCount: count(118),
      synchronizationCount: 1,
      memory: {
        externalBytes: 1_250_000,
        persistentBytes: 4_096,
        stateBytes: programId === "train-step" ? 2_420_000 : 0,
        outputBytes: count(1_310_000),
        workspaceBytes: count(48_200_000),
        transactionBytes: programId === "train-step" ? 1_210_000 : 0,
        peakLiveBytes: count(39_800_000),
        packingOverheadBytes: count(3_100_000)
      },
      compilePhases: [
        { phase: "trace", nanoseconds: 4_200_000 },
        { phase: "autodiff", nanoseconds: 2_900_000 },
        { phase: "legalize", nanoseconds: 1_100_000 },
        { phase: "optimize", nanoseconds: 6_700_000 },
        { phase: "lower", nanoseconds: 3_300_000 },
        { phase: "memory_plan", nanoseconds: 900_000 },
        { phase: "pipelines", nanoseconds: 21_500_000 }
      ]
    }
  }
}

const fourierTarget = (rows: number, cols: number) => {
  const out = new Float32Array(rows * cols)
  for (let c = 0; c < cols; c++) {
    const k = KEY_FREQS[c % KEY_FREQS.length]!
    const phase = c * 0.7
    for (let t = 0; t < rows; t++) out[t * cols + c] = 0.25 * Math.cos((2 * Math.PI * k * t) / P + phase)
  }
  return out
}

const grid = (f: (a: number, b: number) => number): Matrix => {
  const data = new Float32Array(P * P)
  for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) data[a * P + b] = f(a, b)
  return { data, rows: P, cols: P }
}

/** Random-walk state that drifts toward Fourier structure as generalization progresses. */
export const makeProbeSource = (seed: number) => {
  const rng = mulberry32(seed)
  const walk = Float32Array.from({ length: VOCAB * D }, () => gaussian(rng) * 0.088)
  const target = fourierTarget(VOCAB, D)
  const trainMask = Float32Array.from({ length: P * P }, () => (rng() < 0.3 ? 1 : 0))

  return (step: number, grokAt: number, accuracy: Pick<EvalPoint, "trainAcc" | "testAcc">) => {
    const g = grokProgress(step, grokAt)
    for (let i = 0; i < walk.length; i++) walk[i] = walk[i]! * 0.995 + gaussian(rng) * 0.006
    const W_E = walk.map((v, i) => (1 - g) * v + g * target[i]!)
    const grad = Float32Array.from({ length: VOCAB * D }, () => gaussian(rng) * (0.02 * (1 - g) + 0.001))

    const attnTap = new Float32Array(P * P * 36)
    for (let a = 0; a < P; a++) {
      for (let b = 0; b < P; b++) {
        const base = (a * P + b) * 36
        for (let h = 0; h < 4; h++) {
          const k = KEY_FREQS[h]!
          const toA = 0.5 + 0.45 * g * Math.cos((2 * Math.PI * k * (a - b)) / P) + 0.05 * (rng() - 0.5)
          attnTap[base + h * 9 + 6] = toA
          attnTap[base + h * 9 + 7] = 1 - toA
        }
      }
    }

    const heatmaps = [
      heatmapProbe({ name: "embed.W_E", step, sourceShape: [VOCAB, D], matrix: toMatrix(W_E, [VOCAB, D]), scale: "diverging" }),
      heatmapProbe({
        name: "embed.W_E.fourier",
        step,
        sourceShape: [VOCAB, D],
        matrix: fourierNorms(toMatrix(W_E.subarray(0, P * D), [P, D]), P),
        scale: "sequential",
        axes: { rows: "freq", cols: "d" }
      }),
      heatmapProbe({ name: "grad.embed.W_E", step, sourceShape: [VOCAB, D], matrix: toMatrix(grad, [VOCAB, D]), scale: "diverging" }),
      ...[0, 1, 2, 3].map((h) =>
        heatmapProbe({
          name: `attn.h${h}.eq->a`,
          step,
          sourceShape: [P * P, 4, 3, 3],
          matrix: gridChannel({ data: attnTap, shape: [P * P, 4, 3, 3], channel: h * 9 + 2 * 3 + 0, p: P }),
          scale: "probability",
          axes: { rows: "a", cols: "b" }
        })
      ),
      ...Array.from({ length: 8 }, (_, k) => {
        const freq = KEY_FREQS[k % KEY_FREQS.length]!
        return heatmapProbe({
          name: `mlp.n${k}`,
          step,
          sourceShape: [P * P, 512],
          matrix: grid((a, b) => g * Math.max(0, Math.cos((2 * Math.PI * freq * (a + b)) / P + k)) + (1 - g) * gaussian(rng) * 0.4),
          scale: "diverging",
          axes: { rows: "a", cols: "b" }
        })
      }),
      heatmapProbe({
        name: "correct",
        step,
        sourceShape: [P * P],
        matrix: grid((a, b) => (rng() < (trainMask[a * P + b] ? accuracy.trainAcc : accuracy.testAcc) ? 1 : 0)),
        scale: "sequential",
        axes: { rows: "a", cols: "b" },
        highlight: "low",
        levels: { low: "wrong", high: "right" },
        mask: { dimmed: "train", shown: "test", data: trainMask }
      })
    ]

    const histograms = parameters.flatMap(({ name, shape }) => {
      const size = shape.reduce((x, y) => x * y, 1)
      const std = 1 / Math.sqrt(shape[0] ?? 1)
      const param = name === "embed.W_E" ? W_E : Float32Array.from({ length: size }, () => gaussian(rng) * std * (1 - 0.5 * g))
      const gradData = name === "embed.W_E" ? grad : Float32Array.from({ length: size }, () => gaussian(rng) * 0.01 * (1 - g + 0.05))
      return [
        histogramProbe({ name: `hist.${name}`, step, sourceShape: shape, data: param }),
        histogramProbe({ name: `hist.grad.${name}`, step, sourceShape: shape, data: gradData })
      ]
    })

    return [...heatmaps, ...histograms]
  }
}
