import { rgba, theme } from "./theme.ts"

/** Radial accent glow, 0.5 alpha at the center fading to 0, pre-rendered once for `drawImage`. */
export const makeGlowSprite = (px: number, dpr: number) => {
  const size = Math.round(px * dpr)
  const sprite = document.createElement("canvas")
  sprite.width = sprite.height = size
  const ctx = sprite.getContext("2d")
  if (ctx) {
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
    g.addColorStop(0, rgba(theme.accent, 0.5))
    g.addColorStop(1, rgba(theme.accent, 0))
    ctx.fillStyle = g
    ctx.fillRect(0, 0, size, size)
  }
  return sprite
}
