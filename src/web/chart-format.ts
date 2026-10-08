/** Plot height follows width so curve shape survives wide screens. */
export const plotHeight = (width: number) => Math.round(Math.min(320, Math.max(180, width * 0.35)))

const signed = (v: number, text: string) => `${v > 0 ? "+" : "−"}${text}`

const trend = (improving: boolean) => (improving ? "better" : "worse") as "better" | "worse"
const first = { text: "–", trend: "flat" } as const
const flat = { text: "Δ —", trend: "flat" } as const

/**
 * Change since the previous eval: relative for losses (log-scale quantity), absolute for accuracy. A change that
 * rounds to zero at the shown precision reads as flat.
 */
export const evalDelta = (
  { now, before, kind }: { now: number; before: number | undefined; kind: "loss" | "acc" }
) => {
  if (before === undefined || !Number.isFinite(before)) return first
  if (kind === "acc") {
    const d = now - before
    if (Math.abs(d) < 0.0005) return flat
    return { text: `Δ ${signed(d, Math.abs(d).toFixed(3))}`, trend: trend(d > 0) }
  }
  const d = before > 0 ? (now - before) / before : 0
  if (Math.abs(d) < 0.0005) return flat
  return { text: `Δ ${signed(d, `${Math.abs(d * 100).toFixed(1)}%`)}`, trend: trend(d < 0) }
}
