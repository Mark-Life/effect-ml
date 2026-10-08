import { LOGIT_EPS, RAMPS } from "./heatmap-range.ts"
import { hexToRgb, theme } from "./theme.ts"

const vec3 = (hex: string) => `vec3f(${hexToRgb(hex).map((c) => (c / 255).toFixed(4)).join(", ")})`

/** Byte size of `Params`. */
export const PARAMS_BYTES = 32

/** Masked cells turn grey and keep this share of their lightness over the background. */
const MASK_DIM = 0.15
/** Curvature of the log ramp: `t' = ln(1 + K t) / ln(1 + K)`. */
const LOG_K = 40

/**
 * Full-screen triangle; the fragment reads an rg32float texture (value, mask) with textureLoad and maps the value
 * through a LUT row (0 sequential, 1 diverging, 2 two-tone lit high, 3 two-tone lit low). `ramp` is one of
 * `RAMPS`: logit maps the value to log-odds before `lo..hi`, log bends `t` after it. Masked cells are dimmed. `halftone` draws one dot per cell with area proportional to the lit
 * magnitude.
 */
export const colormapWgsl = /* wgsl */ `
struct Params {
  lo: f32,
  hi: f32,
  mode: u32,
  halftone: u32,
  cell: f32,
  width: f32,
  height: f32,
  ramp: u32,
}

@group(0) @binding(0) var values: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var lutTex: texture_2d<f32>;

const BG = ${vec3(theme.bg)};
const MUTED = ${vec3(theme.muted)};
const MASK_DIM = ${MASK_DIM.toFixed(3)};
const LOG_K = ${LOG_K.toFixed(1)};
const LOGIT_EPS = ${LOGIT_EPS.toExponential()};

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var out: VsOut;
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}

fn sampleCell(uv: vec2f) -> vec2f {
  let dims = textureDimensions(values);
  let texel = min(vec2u(clamp(uv, vec2f(0.0), vec2f(1.0)) * vec2f(dims)), dims - vec2u(1u));
  return textureLoad(values, texel, 0).rg;
}

fn dim(color: vec3f, masked: f32) -> vec3f {
  let grey = vec3f(dot(color, vec3f(0.2126, 0.7152, 0.0722)));
  return select(color, mix(BG, grey, MASK_DIM), masked >= 0.5);
}

fn nonFinite(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u;
}

fn toT(v: f32) -> f32 {
  if (params.hi <= params.lo) {
    return 1.0;
  }
  var x = v;
  if (params.ramp == ${RAMPS.logit}u) {
    let q = clamp(v, LOGIT_EPS, 1.0 - LOGIT_EPS);
    x = log(q / (1.0 - q));
  }
  let t = clamp((x - params.lo) / (params.hi - params.lo), 0.0, 1.0);
  if (params.ramp == ${RAMPS.log}u) {
    return log(1.0 + LOG_K * t) / log(1.0 + LOG_K);
  }
  return t;
}

fn lut(t: f32) -> vec3f {
  let i = u32(clamp(t, 0.0, 1.0) * 255.0 + 0.5);
  return textureLoad(lutTex, vec2u(i, params.mode), 0).rgb;
}

fn checker(pos: vec2f) -> vec4f {
  let c = vec2u(pos / 2.0);
  return vec4f(select(BG, MUTED, ((c.x + c.y) & 1u) == 0u), 1.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  if (params.halftone == 0u) {
    let cell = sampleCell(in.uv);
    if (nonFinite(cell.x)) {
      return checker(in.pos.xy);
    }
    return vec4f(dim(lut(toT(cell.x)), cell.y), 1.0);
  }
  let cellIx = floor(in.pos.xy / params.cell);
  let center = (cellIx + 0.5) * params.cell;
  let cell = sampleCell(center / vec2f(params.width, params.height));
  let v = cell.x;
  if (nonFinite(v)) {
    return checker(in.pos.xy);
  }
  let t = toT(v);
  var magnitude = t;
  if (params.mode == 1u) {
    magnitude = abs(t * 2.0 - 1.0);
  } else if (params.mode == 3u) {
    magnitude = 1.0 - t;
  }
  let r = sqrt(magnitude) * 0.5 * params.cell * 0.92;
  let d = distance(in.pos.xy, center);
  let cov = 1.0 - smoothstep(r - 0.75, r + 0.75, d);
  return vec4f(mix(BG, dim(lut(t), cell.y), cov), 1.0);
}
`
