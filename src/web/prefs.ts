/** A boolean setting saved in localStorage, with change listeners. */
export const makePref = (key: string, initial: boolean) => {
  const stored = localStorage.getItem(key)
  let value = stored === null ? initial : stored === "1"
  const listeners = new Set<(v: boolean) => void>()
  const set = (next: boolean) => {
    if (next === value) return
    value = next
    localStorage.setItem(key, next ? "1" : "0")
    for (const l of listeners) l(next)
  }
  return {
    get: () => value,
    set,
    toggle: () => set(!value),
    subscribe: (listener: (v: boolean) => void) => {
      listeners.add(listener)
      return () => void listeners.delete(listener)
    }
  }
}

export type Pref = ReturnType<typeof makePref>

export const ambientPref = makePref("effect-ml.ambient", true)
export const dotsPref = makePref("effect-ml.dots", false)
