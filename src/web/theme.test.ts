import { describe, expect, test } from "bun:test"
import { theme } from "./theme.ts"

const tokens = await Bun.file(new URL("./styles/tokens.css", import.meta.url)).text()
const cssVar = (name: string) => tokens.match(new RegExp(`--${name}:\\s*([^;]+);`))?.[1]?.trim()

describe("theme", () => {
  test.each([
    ["bg", theme.bg],
    ["bg-raise", theme.bgRaise],
    ["fg", theme.fg],
    ["fg-dim", theme.fgDim],
    ["muted", theme.muted],
    ["faint", theme.faint],
    ["accent", theme.accent],
    ["hot", theme.hot],
    ["font-mono", theme.fontMono]
  ])("--%s matches theme.ts", (name, value) => {
    expect(cssVar(name)).toBe(value)
  })
})
