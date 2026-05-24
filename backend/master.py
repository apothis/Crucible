"""Reference-based mastering via Matchering 2.0 — runs LOCALLY on the Mac.

Matches a target track's loudness, frequency response, peak and stereo width to
a *reference* master the user supplies (a pro track they own — e.g. a Halestorm
/ AC-DC / Bon Jovi song). See RESEARCH.md §10a/§10e #2.

Personal-use: the reference is only analysed, never redistributed.
"""


def available():
    try:
        import matchering  # noqa: F401
        return True
    except Exception:
        return False


def master(target_path: str, reference_path: str, out_path: str, bit_depth: int = 16):
    """Master `target_path` toward `reference_path`, writing a WAV to out_path."""
    import matchering as mg

    # keep matchering quiet-ish in the backend log (no crashes on info spam)
    try:
        mg.log(warning_handler=print, info_handler=lambda *a, **k: None,
               show_codename=False)
    except Exception:
        pass

    result = mg.pcm24(out_path) if bit_depth == 24 else mg.pcm16(out_path)
    mg.process(target=target_path, reference=reference_path, results=[result])
    return out_path
