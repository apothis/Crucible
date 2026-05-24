"""Open a plugin's editor and save its VST3 state — run as its OWN PROCESS so
show_editor() is on the main thread with a Cocoa run loop (it fails from a
FastAPI worker thread). Used by the Helix/Kontakt capture endpoints.

Usage:  python -m backend.plugin_capture "<plugin.vst3>" "<out.state>"
Blocks until the user closes the editor window, then writes raw_state.
"""
import sys


def main():
    plugin_path, out_path = sys.argv[1], sys.argv[2]
    import pedalboard as pb
    p = pb.load_plugin(plugin_path)   # instruments (Kontakt) ~25s; effects fast
    p.show_editor()                    # blocks on the main thread until closed
    with open(out_path, "wb") as f:
        f.write(p.raw_state)
    print("saved", out_path, len(p.raw_state), "bytes")


if __name__ == "__main__":
    main()
