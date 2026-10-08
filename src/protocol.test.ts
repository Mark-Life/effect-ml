import type { Runtime } from "@effect-torch/core"
import { describe, expect, test } from "bun:test"
import {
  decodeProbe,
  decodeServerMessage,
  encodeProbe,
  type ExecutableDiagnostics,
  type HeatmapHeader,
  type HistogramHeader
} from "./protocol.ts"

// compile-time: core diagnostics fit the wire schema
const _diagnosticsFit: (d: Runtime.ExecutableDiagnostics) => ExecutableDiagnostics = (d) => d

const heatmap: HeatmapHeader = {
  _tag: "Heatmap",
  name: "embed.W_E",
  step: 100,
  sourceShape: [114, 128],
  rows: 2,
  cols: 3,
  min: -1,
  max: 2,
  scale: "diverging",
  axes: { rows: "a", cols: "b" }
}

const roundTrip = (frame: Uint8Array) =>
  decodeProbe(frame.slice().buffer)

describe("probe frames", () => {
  test("heatmap round trip", () => {
    const payload = Float32Array.from([-1, 0, 1, 2, Number.NaN, 0.5])
    const decoded = roundTrip(encodeProbe(heatmap, payload))
    expect(decoded._tag).toBe("Heatmap")
    expect(decoded.header).toEqual(heatmap)
    expect([...decoded.payload]).toEqual([...payload])
  })

  test("heatmap with a mask plane", () => {
    const header: HeatmapHeader = { ...heatmap, mask: { dimmed: "train", shown: "test" } }
    const values = [0, 1, 1, 0, 1, 1]
    const mask = [1, 0, 0, 1, 0, 0]
    const decoded = roundTrip(encodeProbe(header, Float32Array.from([...values, ...mask])))
    if (decoded._tag !== "Heatmap") throw new Error("expected a heatmap")
    expect([...decoded.payload]).toEqual(values)
    expect([...(decoded.mask ?? [])]).toEqual(mask)
  })

  test("mask header without the mask plane is rejected", () => {
    const header: HeatmapHeader = { ...heatmap, mask: { dimmed: "train", shown: "test" } }
    expect(() => roundTrip(encodeProbe(header, new Float32Array(6)))).toThrow()
  })

  test("histogram round trip", () => {
    const header: HistogramHeader = {
      _tag: "Histogram",
      name: "hist.x",
      step: 1,
      sourceShape: [10],
      bins: 4,
      min: 0,
      max: 1,
      nonFinite: 2
    }
    const payload = Uint32Array.from([1, 2, 3, 4])
    const decoded = roundTrip(encodeProbe(header, payload))
    expect(decoded._tag).toBe("Histogram")
    expect([...decoded.payload]).toEqual([1, 2, 3, 4])
  })

  test("payload starts on a 4-byte boundary for every header length", () => {
    for (let pad = 0; pad < 8; pad++) {
      const header = { ...heatmap, name: `n${"x".repeat(pad)}`, rows: 1, cols: 1 }
      const frame = encodeProbe(header, Float32Array.of(7))
      const headerBytes = new DataView(frame.buffer).getUint32(0, true)
      const offset = frame.byteLength - 4
      expect(offset % 4).toBe(0)
      expect(offset).toBeGreaterThanOrEqual(4 + headerBytes)
      expect(offset - (4 + headerBytes)).toBeLessThan(4)
      expect(frame.subarray(4 + headerBytes, offset).every((b) => b === 0)).toBe(true)
      expect(roundTrip(frame).payload[0]).toBe(7)
    }
  })

  test("rejects a payload that does not match the header", () => {
    expect(() => roundTrip(encodeProbe(heatmap, Float32Array.of(1, 2, 3)))).toThrow(/expects 6/)
  })

  test("rejects a header length past the frame end", () => {
    const frame = encodeProbe(heatmap, new Float32Array(6))
    new DataView(frame.buffer).setUint32(0, frame.byteLength, true)
    expect(() => roundTrip(frame)).toThrow(/does not fit/)
    expect(() => decodeProbe(new ArrayBuffer(2))).toThrow(/too short/)
  })
})

describe("server messages", () => {
  test("decodes tagged messages", () => {
    const message = decodeServerMessage(JSON.parse(JSON.stringify({
      _tag: "Metrics",
      points: [{ step: 1, loss: 2.5, stepMs: 3, memoryBytes: 1024 }]
    })))
    expect(message._tag).toBe("Metrics")
  })

  test("rejects unknown tags", () => {
    expect(() => decodeServerMessage({ _tag: "Nope" })).toThrow()
  })
})
