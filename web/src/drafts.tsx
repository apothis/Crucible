import { createContext, useCallback, useContext, useState, type ReactNode } from "react";

// A per-page draft store. Each tool/form gets a namespace; within it, every
// persisted field is keyed by name. Lives at the App root so switching tabs
// (which unmounts a form) no longer drops its entered values. Project save/load
// serializes this whole store.
export type DraftStore = Record<string, Record<string, unknown>>;

type Ctx = {
  store: DraftStore;
  get: (ns: string, key: string) => unknown;
  set: (ns: string, key: string, v: unknown | ((p: unknown) => unknown)) => void;
  reset: () => void;                 // New project — clear everything
  replaceAll: (s: DraftStore) => void;  // Open project — load a saved store
};

const DraftCtx = createContext<Ctx | null>(null);

export function DraftProvider({ children }: { children: ReactNode }) {
  const [store, setStore] = useState<DraftStore>({});
  const get = useCallback((ns: string, key: string) => store[ns]?.[key], [store]);
  const set = useCallback((ns: string, key: string, v: unknown | ((p: unknown) => unknown)) => {
    setStore((s) => {
      const cur = s[ns] ?? {};
      const next = typeof v === "function" ? (v as (p: unknown) => unknown)(cur[key]) : v;
      return { ...s, [ns]: { ...cur, [key]: next } };
    });
  }, []);
  const reset = useCallback(() => setStore({}), []);
  const replaceAll = useCallback((s: DraftStore) => setStore(s || {}), []);
  return <DraftCtx.Provider value={{ store, get, set, reset, replaceAll }}>{children}</DraftCtx.Provider>;
}

export function useDraftCtx() {
  const c = useContext(DraftCtx);
  if (!c) throw new Error("useDraftCtx must be used inside <DraftProvider>");
  return c;
}

// Drop-in replacement for useState, scoped to a namespace + key. Supports
// functional updaters just like useState. Non-serializable state (File objects,
// object URLs, fetched lists, transient flags) should stay on plain useState.
export function useDrafts(ns: string) {
  const { get, set } = useDraftCtx();
  return {
    use<T>(key: string, initial: T): [T, (v: T | ((p: T) => T)) => void] {
      const cur = get(ns, key);
      const value = (cur === undefined ? initial : cur) as T;
      const setValue = (v: T | ((p: T) => T)) => {
        if (typeof v === "function") {
          set(ns, key, (prev: unknown) => (v as (p: T) => T)(prev === undefined ? initial : (prev as T)));
        } else {
          set(ns, key, v as unknown);
        }
      };
      return [value, setValue];
    },
  };
}
