"""GPU-free ffmpeg video helpers used by the YouTube tab: still/fullbleed/card renders of a
song, the living-cover loop checks (loop_tail_report, seam_loop, trim_frames, seamless_loop)
and audio/video probes. The old music-video script writer and assembly that lived here were
removed 2026-10-27; the rebuilt MV pipeline is docs/MV_H3_REBUILD_PLAN.md. Plain ASCII.
"""
import os
import shutil
import subprocess


def _ffmpeg():
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def _ffprobe():
    return shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"


def audio_duration(path):
    """Duration of an audio file in seconds via ffprobe (0.0 when unreadable)."""
    try:
        r = subprocess.run([_ffprobe(), "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", path], capture_output=True, text=True, timeout=60)
        return float((r.stdout or "").strip())
    except (ValueError, subprocess.SubprocessError, OSError):
        return 0.0


def _viz_filter(viz, width, height, position, scale):
    """Audio-reactive overlay chain for still_video. The visualiser is generated from the
    ACTUAL samples (ffmpeg showwaves/showfreqs/showspectrum, sample-accurate), keyed over
    the cover art at one of the 12 lettering slots. Placement fractions mirror
    wordmark._place so the visualiser grid in the UI means the same thing as the
    title/band-name grids. `scale` = visualiser width as a fraction of the canvas."""
    vw = int(round(width * max(0.15, min(0.60, float(scale or 0.33))))) // 2 * 2
    vh = int(round(vw * 0.28)) // 2 * 2
    parts = str(position or "bottom-right").split("-")
    vert, horiz = parts[0], (parts[1] if len(parts) > 1 else "center")
    if horiz == "left":
        x = round(width * 0.06)
    elif horiz == "right":
        x = round(width * 0.94) - vw
    else:
        x = (width - vw) // 2
    if vert == "header":
        y = round(height * 0.055)
    elif vert == "top":
        y = round(height * 0.24)
    elif vert == "middle":
        y = (height - vh) // 2
    else:
        y = round(height * 0.93) - vh
    if viz == "bars":
        gen = (f"showfreqs=s={vw}x{vh}:mode=bar:ascale=cbrt:fscale=log:"
               f"win_size=2048:colors=0xffc27a|0xff7a3c")
    elif viz == "spectro":
        # log frequency axis: the music lives in the low/mid bands, a linear axis
        # squeezes it all into a thin strip at the bottom of the box
        gen = (f"showspectrum=s={vw}x{vh}:slide=scroll:mode=combined:color=fire:"
               f"scale=cbrt:fscale=log:legend=0:fps=25")
    else:  # "waves"
        # draw=full: the default draw=scale accumulates sample intensity and shifts
        # the hue (a plain orange came out with a mint-green core)
        gen = f"showwaves=s={vw}x{vh}:mode=cline:rate=25:draw=full:colors=0xff9a4d"
    # waves draws MONO: overlapping stereo channels accumulate the same way, and
    # the mono mix is the classic centered waveform anyway
    layout = "mono" if viz not in ("bars", "spectro") else "stereo"
    # the generators draw on black; colorkey makes that transparent so only the
    # waveform/bars sit over the art (soft blend edge to avoid a dark halo)
    return (f"[1:a]aformat=channel_layouts={layout},{gen}[vz0];"
            f"[vz0]colorkey=0x000000:0.10:0.30[vz];"
            f"[bg][vz]overlay={max(0, x)}:{max(0, y)}[v]")


def still_video(image_path, audio_path, out_path, width=1920, height=1080, fps=1,
                title="", artist="", viz="", viz_position="bottom-right", viz_scale=0.33,
                progress_cb=None):
    """One cover still + the full song -> an upload-ready static MP4 (the YouTube
    "art track" shape). GPU-free ffmpeg on the Mac: the image loops at `fps` (1 fps
    makes the video stream nearly free next to the audio), scaled and padded onto a
    width x height canvas, x264 -tune stillimage, AAC 320k, +faststart. The length
    comes from ffprobe on the audio and is passed as an explicit -t: the classic
    -loop/-shortest pairing can keep muxing past the audio end (interleave quirk),
    and an explicit duration sidesteps it.
    viz ("waves"|"bars"|"spectro") overlays an audio-reactive visualiser, which turns
    the output into a real 25 fps encode (minutes of CPU instead of seconds - pass
    progress_cb(seconds_done, total_seconds) and run it off-thread)."""
    dur = audio_duration(audio_path)
    if dur <= 0:
        raise ValueError("could not read the audio duration")
    base_vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    cmd = [_ffmpeg(), "-y", "-loglevel", "error"]
    if progress_cb:
        cmd += ["-nostats", "-progress", "pipe:1"]
    if viz:
        cmd += ["-loop", "1", "-framerate", "25", "-i", image_path,
                "-i", audio_path,
                "-filter_complex",
                f"[0:v]{base_vf}[bg];" + _viz_filter(viz, width, height, viz_position, viz_scale),
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", "25"]
    else:
        cmd += ["-loop", "1", "-framerate", str(fps), "-i", image_path,
                "-i", audio_path, "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage",
                "-pix_fmt", "yuv420p", "-vf", base_vf, "-r", str(fps)]
    cmd += ["-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart"]
    # container metadata: YouTube ignores it, but every player shows it, and it costs nothing
    if title:
        cmd += ["-metadata", f"title={title}"]
    if artist:
        cmd += ["-metadata", f"artist={artist}"]
    cmd += ["-t", f"{dur:.3f}", out_path]
    if not progress_cb:
        subprocess.run(cmd, check=True)
        return out_path
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("out_time_us="):
            try:
                progress_cb(int(line.split("=", 1)[1]) / 1e6, dur)
            except ValueError:
                pass
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {(proc.stderr.read() or '')[-400:]}")
    return out_path


def _run_progress(cmd, dur, progress_cb):
    """Run an ffmpeg command; with progress_cb, stream -progress and report seconds done."""
    if not progress_cb:
        subprocess.run(cmd, check=True)
        return
    cmd = cmd[:1] + ["-nostats", "-progress", "pipe:1"] + cmd[1:]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("out_time_us="):
            try:
                progress_cb(int(line.split("=", 1)[1]) / 1e6, dur)
            except ValueError:
                pass
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {(proc.stderr.read() or '')[-400:]}")


def loop_tail_report(path, overrender=0):
    """Measure a living-cover loop's tail behavior (the end-pin failure modes, measured
    2026-09-02 on The Endless Sea takes): decode to small grayscale, mean abs diff between
    consecutive frames = motion. Returns
      {frames, mid, tail_ratio, frozen_from, trim_frame}
    mid        = mean motion across the middle half (the reference speed)
    tail_ratio = final-second motion / mid  (< ~0.8 reads as an ease-in hesitation)
    frozen_from= first frame of a terminal run with motion < 15% of mid (None if none)
    trim_frame = suggested exclusive cut BEFORE the frozen run, but only while the cut
                 frame still lands within ~1.3 frame-steps of frame 1 (deeper cuts turn
                 the pause into a visible position jump - measured, do not trim harder).
    overrender = extra frames the take was deliberately rendered long (the same-still loop
                 recipe, measured 2026-09-06). The FL2VA model decelerates onto its last
                 anchor and parks, so an over-rendered take is cut at its LOCK POINT: the
                 first frame after which motion stays under 25% of the clip's median to the
                 end. The trim cap grows by the over-render; the wrap-step guard still applies
                 (the lock frame already matches frame 1 - measured 1.1x a normal step)."""
    import numpy as np
    W, H = 320, 180
    p = subprocess.run([_ffmpeg(), "-v", "error", "-i", path,
                        "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    buf = np.frombuffer(p.stdout, dtype=np.uint8)
    n = len(buf) // (W * H)
    if n < 50:
        raise ValueError("clip too short to analyze")
    fr = buf[:n * W * H].reshape(n, H, W).astype(np.int16)
    diffs = np.abs(np.diff(fr, axis=0))
    motion = diffs.mean(axis=(1, 2))
    # localized motion: the 99th-percentile pixel change per frame pair. Fine motion
    # (dust motes, candle flicker, a breath) barely moves the MEAN but lights this up;
    # a frame is only "frozen" when BOTH say nothing moved. Added 2026-09-05 after the
    # mean-only rule judged an OpenArt clip (real but subtle motion, mid 0.13) frozen
    # from frame 1 and the auto-trim amputated it to a single frame.
    local = np.percentile(diffs.reshape(diffs.shape[0], -1), 99, axis=1)
    mid = float(motion[len(motion) // 4:3 * len(motion) // 4].mean())
    tail = motion[-24:]
    # freeze threshold: relative to the clip's own pace, with a floor that SCALES with
    # the clip (capped at 0.30) - a flat floor above a subtle clip's whole range calls
    # everything frozen
    thresh = max(0.15 * mid, min(0.30, 0.5 * mid))
    frozen_from = None
    i = n - 1
    while i >= 1 and motion[i - 1] < thresh and local[i - 1] < 6:
        frozen_from = i
        i -= 1
    # a frozen TAIL is a tail: a run longer than ~1s is a stalled clip (reported via
    # idle_frac), never something to cut - trimming can remove at most 24 frames
    # (plus the deliberate over-render, which exists to be cut away)
    max_trim = 24 + max(0, int(overrender or 0))
    if frozen_from is not None and frozen_from < n - max_trim:
        frozen_from = None
    trim_frame = None
    f0 = fr[0]
    if overrender:
        med = float(np.median(motion)) or mid
        # EARLIEST-CLOSE cut (measured 2026-09-06 on a turbo take: the picture matches frame 1
        # within a normal frame step from ~f240 while motion is still alive, but waiting for
        # motion to die left 0.6s of near-freeze in): cut at the first frame of the trim window
        # whose jump back to frame 1 is no bigger than an ordinary frame step, and which stays
        # that close for the next 6 frames (so a coincidental early match cannot fool it).
        # The parked tail after it is exactly the settle we over-rendered to throw away.
        guard = max(1.3 * med, 1.0)
        wrap = np.array([np.abs(fr[c] - f0).mean() for c in range(n)])
        first_close = None
        for c in range(max(1, n - max_trim), n - 6):
            if wrap[c] <= guard and wrap[c:c + 6].max() <= guard:
                first_close = c
                break
        if first_close is not None:
            trim_frame = first_close
            frozen_from = frozen_from if frozen_from is not None else first_close
        else:
            # fallback: the motion lock point on the MEAN only (the localized p99 test stays
            # for plain takes: a parked tail still flickers flames/snow, which would mask it)
            lock = 1
            for i in range(len(motion) - 1, -1, -1):
                if motion[i] > 0.25 * med:
                    lock = i + 1
                    break
            if n - max_trim <= lock < n and np.abs(fr[lock] - f0).mean() <= 1.3 * mid:
                trim_frame = lock
                frozen_from = frozen_from if frozen_from is not None else lock
        # the model also eases OUT of the first anchor (~0.5s of near-static frames at the
        # head): the seam treatment drops them, so report where the motion reaches half pace
        roll = np.convolve(motion, np.ones(12) / 12, mode="valid")
        head_trim = min(24, len(roll))
        for i in range(min(24, len(roll))):
            if roll[i] >= 0.5 * med:
                head_trim = i
                break
    if frozen_from is not None and trim_frame is None:
        cut = frozen_from
        # walk back only while the wrap step stays ~one normal frame of motion
        while cut > n - max_trim and np.abs(fr[cut - 1] - f0).mean() > 1.3 * mid:
            cut -= 1
        if cut >= n - max_trim and np.abs(fr[cut - 1] - f0).mean() <= 1.3 * mid:
            trim_frame = cut
    # interior stalls (arrive-early-and-idle) are NOT trimmable - cutting one makes a
    # jump - but they must be reported: fraction of the last 2s spent below threshold
    tail2 = motion[-50:]
    idle_frac = float((tail2 < thresh).mean())
    # a wrap that never closes: last frame vs frame 1 in units of the clip's own pace
    wrap_step = float(np.abs(fr[-1] - f0).mean())
    out = {"frames": n, "mid": round(mid, 2),
           "tail_ratio": round(float(tail.mean()) / mid, 2) if mid else 1.0,
           "idle_frac": round(idle_frac, 2),
           "wrap_step": round(wrap_step, 2),
           "wrap_step_ratio": round(wrap_step / mid, 1) if mid else 0.0,
           "frozen_from": frozen_from, "trim_frame": trim_frame}
    if overrender:
        out["head_trim"] = int(head_trim)
    return out


def seam_loop(in_path, out_path, head, tail, fade=0.5):
    """The STANDARD seam treatment for over-rendered same-still H3 loops (user-chosen
    2026-09-06, 'B'): keep frames [head, tail) - dropping the eased-out static head and the
    eased-in parked tail - then crossfade the moving tail into the moving head over `fade`
    seconds. Measured on the viking take: motion through the seam stays at the clip's normal
    pace (1.2-1.5 vs median 1.9) with a 0.8x-step jump, where a plain cut left ~0.5s of
    near-freeze (0.07-0.13). Both sides of the dissolve are MOVING, which is what hides it;
    this is not the 1.5s still-into-still crossfade used for background loops."""
    body = out_path + ".body.mp4"
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", in_path,
                    "-vf", f"select='between(n,{int(head)},{int(tail) - 1})',setpts=PTS-STARTPTS",
                    "-vsync", "0", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", body],
                   check=True)
    try:
        seamless_loop(body, out_path, fade=fade)
    finally:
        try:
            os.remove(body)
        except OSError:
            pass
    return out_path


def trim_frames(in_path, out_path, end_frame):
    """Re-encode keeping frames [0, end_frame) - used to drop a loop's frozen tail."""
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", in_path,
                    "-vf", f"select='lt(n,{int(end_frame)})'", "-vsync", "0", "-an",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out_path],
                   check=True)
    return out_path


def seamless_loop(in_path, out_path, fade=1.5):
    """Crossfade a clip's head into its own tail -> a seamlessly loopable clip (duration
    shrinks by `fade`). The output's last frame flows straight into its first: verified
    invisible on particle fields (embers/fog backgrounds); NOT needed for H3 FLF loops,
    which pin their own endpoints. Video only (loops are muxed with the song later)."""
    dur = _video_duration(in_path)
    if dur <= 2 * fade + 0.5:
        raise ValueError(f"clip too short ({dur:.1f}s) for a {fade}s crossfade loop")
    off = round((dur - fade) - fade, 3)
    fc = (f"[0:v]split[b][p];[p]trim=0:{fade},setpts=PTS-STARTPTS[fin];"
          f"[b]trim={fade},setpts=PTS-STARTPTS[rest];"
          f"[rest][fin]xfade=transition=fade:duration={fade}:offset={off}[v]")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", in_path,
                    "-filter_complex", fc, "-map", "[v]", "-an",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out_path],
                   check=True)
    return out_path


def _video_duration(path):
    out = subprocess.run([_ffmpeg().replace("ffmpeg", "ffprobe"), "-v", "error",
                          "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, timeout=30)
    return float(out.stdout.strip().split()[0])


def _loop_av_cmd(dur, title, artist, out_path):
    """Shared encode tail for the loop-video assemblers."""
    cmd = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-r", "25",
           "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart"]
    if title:
        cmd += ["-metadata", f"title={title}"]
    if artist:
        cmd += ["-metadata", f"artist={artist}"]
    return cmd + ["-t", f"{dur:.3f}", out_path]


def fullbleed_video(loop_path, audio_path, plate_path, out_path,
                    width=1920, height=1080, title="", artist="", progress_cb=None):
    """Living-cover video, layout C: the loop clip repeats for the whole song, scaled and
    padded to the canvas, with the transparent lettering plate (wordmark/title) on top."""
    dur = audio_duration(audio_path)
    if dur <= 0:
        raise ValueError("could not read the audio duration")
    fc = (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
          f"[bg][2:v]overlay=0:0[v]")
    cmd = [_ffmpeg(), "-y", "-loglevel", "error",
           "-stream_loop", "-1", "-i", loop_path,
           "-i", audio_path, "-i", plate_path,
           "-filter_complex", fc, "-map", "[v]", "-map", "1:a"]
    _run_progress(cmd + _loop_av_cmd(dur, title, artist, out_path), dur, progress_cb)
    return out_path


def card_video(bg_loop_path, card_path, card_is_video, audio_path, plate_path, out_path,
               layout, width=1920, height=1080, title="", artist="",
               viz=True, progress_cb=None, endcard_plate_path=None, endcard_secs=15.0,
               wave_color="0xff9a4d"):
    """Card-visualizer video, layout B (the user's pick): background loop full-frame,
    the cover (living-cover loop or still) inside the card, the card_plate PNG
    (border + band + title) on top, and an audio-reactive waveform in the text column.
    `layout` = wordmark.CARD_LAYOUT scaled to the canvas by the caller.
    endcard_plate_path (opt-in): for the final `endcard_secs` the title plate and the
    waveform are switched off and this plate (wordmark + YouTube end-screen panels, see
    wordmark.endcard_plate) is shown instead - the landing zone for Studio's end-screen
    elements. The loop and background keep playing; the song plays out underneath."""
    dur = audio_duration(audio_path)
    if dur <= 0:
        raise ValueError("could not read the audio duration")
    L = layout
    cmd = [_ffmpeg(), "-y", "-loglevel", "error"]
    if bg_loop_path.lower().endswith((".png", ".jpg", ".jpeg")):
        cmd += ["-loop", "1", "-framerate", "25", "-i", bg_loop_path]   # static backdrop (modern look)
    else:
        cmd += ["-stream_loop", "-1", "-i", bg_loop_path]
    if card_is_video:
        cmd += ["-stream_loop", "-1", "-i", card_path]
    else:
        cmd += ["-loop", "1", "-framerate", "25", "-i", card_path]
    cmd += ["-i", audio_path, "-i", plate_path]
    ec = None
    if endcard_plate_path and endcard_secs and dur > float(endcard_secs) + 10:
        ec = max(0.0, dur - float(endcard_secs))       # the end card starts here
        cmd += ["-i", endcard_plate_path]
    body_on = f":enable='lt(t,{ec:.3f})'" if ec is not None else ""
    fc = (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},setsar=1[bg];"
          f"[1:v]scale={L['card_w']}:{L['card_h']}:force_original_aspect_ratio=decrease:"
          f"force_divisible_by=2,"
          f"pad={L['card_w']}:{L['card_h']}:(ow-iw)/2:(oh-ih)/2,setsar=1[card];"
          f"[bg][card]overlay={L['card_x']}:{L['card_y']}[b1];"
          f"[b1][3:v]overlay=0:0{body_on}[b2];")
    if ec is not None:
        fc += f"[b2][4:v]overlay=0:0:enable='gte(t,{ec:.3f})'[b3];"
        last = "b3"
    else:
        last = "b2"
    if viz:
        fc += (f"[2:a]aformat=channel_layouts=mono,"
               f"showwaves=s={L['wave_w']}x{L['wave_h']}:mode=cline:rate=25:draw=full:"
               f"colors={wave_color}[vz0];[vz0]colorkey=0x000000:0.10:0.30[vz];"
               f"[{last}][vz]overlay={L['wave_x']}:{L['wave_y']}{body_on}[v]")
    else:
        fc += f"[{last}]null[v]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "2:a"]
    _run_progress(cmd + _loop_av_cmd(dur, title, artist, out_path), dur, progress_cb)
    return out_path
