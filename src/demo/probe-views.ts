import { fourierNorms, gridChannel, heatmapProbe, histogramProbe, toMatrix } from "../probes.ts"
import type { EvalPoint } from "../protocol.ts"

export interface Read {
  readonly data: Float32Array
  readonly shape: ReadonlyArray<number>
}

export interface ProbeRead {
  readonly step: number
  readonly p: number
  readonly heads: number
  readonly seq: number
  readonly paramNames: ReadonlyArray<string>
  readonly params: ReadonlyArray<Read>
  readonly grads: ReadonlyArray<Read>
  readonly perExample: Float32Array
  readonly correct: Float32Array
  readonly trainMask: Float32Array
  readonly pattern: Read
  readonly hidden: Read
}

const MLP_NEURONS = 8

/** Splits per-example loss and accuracy by the train mask. */
export const evalPoint = (
  { step, perExample, correct, trainMask }: Pick<ProbeRead, "step" | "perExample" | "correct" | "trainMask">,
  probeMs: number
): EvalPoint => {
  let trainLoss = 0, testLoss = 0, trainAcc = 0, testAcc = 0, trainN = 0, testN = 0
  for (let i = 0; i < perExample.length; i++) {
    if (trainMask[i] === 1) {
      trainLoss += perExample[i]!
      trainAcc += correct[i]!
      trainN++
    } else {
      testLoss += perExample[i]!
      testAcc += correct[i]!
      testN++
    }
  }
  const avg = (sum: number, n: number) => (n === 0 ? 0 : sum / n)
  return {
    _tag: "Eval",
    step,
    trainLoss: avg(trainLoss, trainN),
    testLoss: avg(testLoss, testN),
    trainAcc: avg(trainAcc, trainN),
    testAcc: avg(testAcc, testN),
    probeMs
  }
}

/** Heatmap and histogram frames for one probe pass. */
export const probeFrames = (r: ProbeRead) => {
  const { step, p, heads, seq } = r
  const grid = { rows: "a", cols: "b" } as const
  const embedIndex = r.paramNames.indexOf("embed.W_E")
  const embed = r.params[embedIndex]
  const embedGrad = r.grads[embedIndex]
  const d = embed?.shape[1] ?? 0

  const embedViews = embed && embedGrad
    ? [
      heatmapProbe({
        name: "embed.W_E",
        step,
        sourceShape: embed.shape,
        matrix: toMatrix(embed.data, embed.shape),
        scale: "diverging",
        axes: { rows: "token", cols: "d" }
      }),
      heatmapProbe({
        name: "embed.W_E.fourier",
        step,
        sourceShape: embed.shape,
        matrix: fourierNorms(toMatrix(embed.data.subarray(0, p * d), [p, d]), p),
        scale: "sequential",
        axes: { rows: "freq", cols: "d" }
      }),
      heatmapProbe({
        name: "grad.embed.W_E",
        step,
        sourceShape: embedGrad.shape,
        matrix: toMatrix(embedGrad.data, embedGrad.shape),
        scale: "diverging",
        axes: { rows: "token", cols: "d" }
      })
    ]
    : []

  // pattern [N, H, q, k]: last query (the "=" token) attending to key 0 (operand a)
  const attention = Array.from({ length: heads }, (_, h) =>
    heatmapProbe({
      name: `attn.h${h}.eq->a`,
      step,
      sourceShape: r.pattern.shape,
      matrix: gridChannel({ ...r.pattern, channel: h * seq * seq + (seq - 1) * seq, p }),
      scale: "probability",
      axes: grid
    }))

  const neurons = Array.from({ length: MLP_NEURONS }, (_, k) =>
    heatmapProbe({
      name: `mlp.n${k}`,
      step,
      sourceShape: r.hidden.shape,
      matrix: gridChannel({ ...r.hidden, channel: k, p }),
      scale: "sequential",
      axes: grid
    }))

  const correctProbe = heatmapProbe({
    name: "correct",
    step,
    sourceShape: [p * p],
    matrix: gridChannel({ data: r.correct, shape: [p * p], channel: 0, p }),
    scale: "sequential",
    axes: grid,
    highlight: "low",
    levels: { low: "wrong", high: "right" },
    mask: { dimmed: "train", shown: "test", data: r.trainMask }
  })
  // fixed 0..1 range so an all-correct grid renders as 1, not as a degenerate range
  const correct = { ...correctProbe, header: { ...correctProbe.header, min: 0, max: 1 } }

  const histograms = r.paramNames.flatMap((name, i) => {
    const param = r.params[i]
    const grad = r.grads[i]
    return param && grad
      ? [
        histogramProbe({ name: `hist.${name}`, step, sourceShape: param.shape, data: param.data }),
        histogramProbe({ name: `hist.grad.${name}`, step, sourceShape: grad.shape, data: grad.data })
      ]
      : []
  })

  return [...embedViews, ...attention, ...neurons, correct, ...histograms]
}
