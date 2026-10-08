import { describe, expect, test } from "bun:test"
import { fitLabel } from "./graph-view.ts"

describe("fitLabel", () => {
  test("whole label when it fits", () => expect(fitLabel("matmul", 60)).toBe("matmul"))
  test("cuts with an ellipsis", () => expect(fitLabel("cross_entropy", 52)).toBe("cross_…"))
  test("empty when under two characters fit", () => expect(fitLabel("add", 20)).toBe(""))
  test("caps long labels", () => expect(fitLabel("x".repeat(40), 1000).length).toBe(23))
})
