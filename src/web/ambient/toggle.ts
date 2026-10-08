import { ambientPref } from "../prefs.ts"

/** Binds the `AMBIENT ON/OFF` chips and the `a` key to the ambient preference. */
export const bindAmbientToggle = (chips: ReadonlyArray<HTMLElement>) => {
  const paint = (on: boolean) => {
    for (const chip of chips) {
      chip.textContent = on ? "ambient on" : "ambient off"
      chip.dataset["state"] = on ? "on" : "off"
      chip.setAttribute("aria-pressed", String(on))
    }
  }
  paint(ambientPref.get())
  ambientPref.subscribe(paint)
  for (const chip of chips) chip.addEventListener("click", ambientPref.toggle)
  window.addEventListener("keydown", (e) => {
    if (e.key !== "a" || e.metaKey || e.ctrlKey || e.altKey) return
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return
    ambientPref.toggle()
  })
}
