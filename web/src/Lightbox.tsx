import { useEffect, useState } from "react";

// Click-to-enlarge image viewer. Videos already get native fullscreen via <video controls>;
// stills did not, so any <img> can call openLightbox(url) to show it full-size over a dark
// overlay. Click the backdrop or press Escape to close. One <Lightbox/> is mounted at the
// app root; openLightbox is a module-level opener so any component can trigger it without
// threading props or importing App (which would be a circular import).
let setOpen: ((url: string | null) => void) | null = null;

export function openLightbox(url: string) {
  setOpen?.(url);
}

export function Lightbox() {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    setOpen = setUrl;
    return () => { setOpen = null; };
  }, []);
  useEffect(() => {
    if (!url) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setUrl(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [url]);
  if (!url) return null;
  return (
    <div
      onClick={() => setUrl(null)}
      className="fixed inset-0 z-[100] flex cursor-zoom-out items-center justify-center bg-black/80 p-4"
    >
      <img
        src={url}
        alt=""
        onClick={(e) => e.stopPropagation()}
        className="max-h-full max-w-full rounded-lg shadow-2xl"
      />
    </div>
  );
}
