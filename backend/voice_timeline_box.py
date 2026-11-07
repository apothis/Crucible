"""Speaker fingerprints for stretches of a vocal stem (who sings when), run on the GPU box.

Deployed by the Mac (fs/write) into ComfyUI_windows_portable\\ and run by the :5080 helper's
POST /voice/timeline under ComfyUI's own embedded python (onnxruntime, torch, torchaudio are
already there - nothing to install). One shot: loads the model, embeds every segment, prints
one JSON line, exits, so nothing stays in memory afterwards.

Model: WeSpeaker ResNet34 (VoxCeleb, CC-BY-4.0), voxceleb_resnet34_LM.onnx. Features follow the
model author's own infer_onnx.py: 16 kHz mono, waveform * 2^15, kaldi fbank 80 bins, 25 ms frame,
10 ms shift, dither 0, hamming window, no energy, then per-segment mean normalisation (CMN).

Usage: python voice_timeline_box.py <audio.wav> <segments.json> <model.onnx> <out.json>
  audio.wav: 16-bit PCM WAV, 16 kHz mono (the Mac resamples the stem before upload)
  segments.json: {"segments": [[start_s, end_s], ...]}
Writes {"dim", "provider", "embeddings": [[...], ...]} (one per segment, L2-normalised;
null where a segment was too short) to out.json and prints a short summary line.
"""
import json
import sys
import time
import wave

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi

SR = 16000
MIN_SECONDS = 0.5


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"need 16 kHz mono 16-bit wav, got {w.getframerate()} Hz "
                             f"{w.getnchannels()} ch {8 * w.getsampwidth()} bit")
        raw = w.readframes(w.getnframes())
    # int16 samples are already on the 2^15 scale the model was trained with
    return np.frombuffer(raw, dtype="<i2").astype(np.float32)


def main():
    audio_path, seg_path, model_path, out_path = sys.argv[1:5]
    t0 = time.time()
    pcm = read_wav(audio_path)
    with open(seg_path) as f:
        segs = json.load(f)["segments"]
    want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in want if p in ort.get_available_providers()]
    sess = ort.InferenceSession(model_path, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    embs, dim = [], None
    for s, e in segs:
        a, b = int(max(0.0, float(s)) * SR), int(float(e) * SR)
        chunk = pcm[a:b]
        if len(chunk) < MIN_SECONDS * SR:
            embs.append(None)
            continue
        wav = torch.from_numpy(chunk.copy()).unsqueeze(0)
        mat = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                          sample_frequency=SR, window_type="hamming", use_energy=False)
        mat = mat - torch.mean(mat, dim=0)
        v = sess.run([out_name], {in_name: mat.unsqueeze(0).numpy()})[0][0].astype(np.float64)
        v = v / (np.linalg.norm(v) + 1e-9)
        dim = len(v)
        embs.append([round(float(x), 5) for x in v])
    with open(out_path, "w") as f:
        json.dump({"dim": dim, "provider": sess.get_providers()[0], "embeddings": embs}, f)
    print(json.dumps({"out": out_path, "segments": len(segs), "embedded": sum(x is not None for x in embs),
                      "dim": dim, "provider": sess.get_providers()[0],
                      "seconds": round(time.time() - t0, 2)}))


if __name__ == "__main__":
    main()
