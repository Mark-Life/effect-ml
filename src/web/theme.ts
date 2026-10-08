/** Palette for canvas, uPlot and WGSL. Mirrors `styles/tokens.css`; `theme.test.ts` keeps them equal. */
export const theme = {
  bg: "#0a0a0a",
  bgRaise: "#0f0f0f",
  fg: "#e6e3df",
  fgDim: "#a3a09c",
  muted: "#858280",
  faint: "#2e2c2a",
  accent: "#e0503a",
  hot: "#fff1e8",
  /** Top of the heatmap colormaps: a light ember, so saturated cells glow without blowing out to white. */
  heatTop: "#ea7c62",
  hair: "rgba(255,255,255,0.12)",
  hair2: "rgba(255,255,255,0.06)",
  fontMono: `"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace`
} as const

export interface Stop {
  readonly at: number
  readonly color: string
}

/** Sequential map: lightness rises at every stop; the bulk stays dark rust, accent and light ember only at the top. */
export const emberStops: ReadonlyArray<Stop> = [
  { at: 0, color: theme.bg },
  { at: 0.3, color: "#1f0b07" },
  { at: 0.55, color: "#3d0f08" },
  { at: 0.72, color: "#6e1a0e" },
  { at: 0.86, color: "#b0321f" },
  { at: 0.95, color: theme.accent },
  { at: 1, color: theme.heatTop }
]

/** Diverging map on [-1, 1]: steel below zero, ember above, black at zero; both sides match in OKLab lightness. */
export const divergingStops: ReadonlyArray<Stop> = [
  { at: -1, color: "#83a6c4" },
  { at: -0.85, color: "#638caf" },
  { at: -0.5, color: "#253f55" },
  { at: 0, color: "#121212" },
  { at: 0.5, color: "#6e1a0e" },
  { at: 0.85, color: theme.accent },
  { at: 1, color: theme.heatTop }
]

/** Two-level maps: `high` lights the max cells, `low` lights the min cells; the viewer lights the rarer level. */
export const twoToneStops = {
  high: { lo: theme.faint, hi: theme.accent },
  low: { lo: theme.accent, hi: theme.faint }
} as const

export const hexToRgb = (hex: string) => {
  const n = Number.parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255] as const
}

export const rgba = (hex: string, alpha: number) => {
  const [r, g, b] = hexToRgb(hex)
  return `rgba(${r},${g},${b},${alpha})`
}

export const mixHex = (a: string, b: string, t: number) => {
  const x = hexToRgb(a)
  const y = hexToRgb(b)
  const at = (i: 0 | 1 | 2) => Math.round(x[i] + (y[i] - x[i]) * t)
  return [at(0), at(1), at(2)] as const
}

export const monoFont = (px: number, weight = 400) => `${weight} ${px}px ${theme.fontMono}`

export const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches
