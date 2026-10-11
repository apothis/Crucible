# "Neutral open master" preset (Mixea Medium Neutral, reverse-engineered)

Master tab -> Auto -> Quick preset "Neutral open master". Built 2026-10-11 from a measured
A/B of a Suno take ("Even the Rain", 4:44, -11.5 LUFS) against the Mixea (DistroKid)
"Medium Neutral" master of the same file. Everything below is MEASURED on that pair
(.mvwork/etr_master/analyse.py, compare3.py, band_ratio.py, fit_eq.py) unless marked [H].

## What Mixea did (measured)
| | Suno | Mixea |
|---|---|---|
| Integrated loudness | -11.5 LUFS | -11.3 LUFS |
| Loudness range | 8.7 LU | 7.6 LU |
| Crest factor | 14.7 dB | 14.9 dB |
| True peak | -0.15 dBTP | +0.45 dBTP (sample peak -0.1 dBFS) |
| L/R correlation | 0.70 | 0.57 |
| Side/mid energy | -7.5 dB | -5.6 dB |

1. Static "smile" EQ: +1.5..2 dB 50-160 Hz (peak ~90 Hz), broad -3..-4.5 dB dip 300 Hz-1.3 kHz
   deepest at 800 Hz, +2..3 dB 3-10 kHz peaking near 7 kHz, nothing above 16 kHz.
2. Three-band levelling, no loudness push: quiet passages +2 dB, loud passages +0 dB; per-band
   out/in slopes on 0.4 s windows (split 250 Hz / 2.5 kHz) 0.93 / 0.76 / 0.82, slow release
   (~1.5 s: the bridge entry after a loud chorus lands 3 dB LOWER than the source).
3. Broadband stereo widener: side +2 dB in every band, sub bass included.
4. Sample-peak-only limiting: true peaks go 0.45 dB over. Not copied.

## What the preset does
- Tone curve `neutral` (backend/master.py TONE_CURVES): least-squares fit of pedalboard filters
  to the measured curve, 0.19 dB rms error 40 Hz-12 kHz in isolation, 0.40 dB on the rendered
  master. Values: highpass 25 Hz; low shelf 300 Hz +3.28 dB; bells 302 Hz -3.20 dB Q 0.97,
  898 Hz -3.80 dB Q 1.23, 6348 Hz +3.59 dB Q 0.47. The shelf+bell pair at 300 Hz is what the
  fit needed for the asymmetric dip.
- Leveller (`leveller` 0..1, default 0 = off): three zero-phase Butterworth bands
  (power-complementary, sum back exactly flat), each band's 0.4 s RMS vs its own track RMS,
  gain = amount*(1-1/ratio)*(ref-level), clamp +-6 dB, attack 50 ms, release 1.5 s. Ratios
  (LEVELLER_BANDS) are CALIBRATED so the achieved slopes match: 1.15 / 1.41 / 1.39 nominal ->
  achieved 0.923 / 0.755 / 0.817 vs Mixea 0.928 / 0.759 / 0.823. Not a limiter.
- Glue compressor OFF (`glue=false`); Mixea's crest factor was unchanged.
- Width 1.25 (+1.9 dB side) with bass-mono 60 Hz (Mixea widened the subs too; 60 Hz keeps
  the sub centred and matches above 100 Hz; set 0 for an exact copy).
- Loudness "keep" (target_lufs 0 = the source's own integrated LUFS): no push.
- True-peak ceiling -1.0 dBTP (the auto chain's ceiling check now uses 4x-oversampled true
  peak; before 2026-10-11 it used sample peak and a "-1 dBTP" master could land at 0 dBTP).

## Results (Even the Rain, final run)
| | Suno | Mixea | Preset |
|---|---|---|---|
| LUFS | -11.53 | -11.30 | -11.35 |
| Quiet lift (ST < -20 LUFS) | | +2.07 dB | +2.11 dB |
| Loud change (ST > -10) | | +0.03 dB | -0.45 dB |
| EQ rms error vs Mixea | | | 0.40 dB |
| Side/mid | -7.5 | -5.6 | -5.9 dB |
| True peak | -0.15 | +0.43 | -1.00 dBTP |
| Crest factor | 14.7 | 14.9 | 12.4 dB |

The one real remaining difference is the crest factor: the EQ boosts lift the peaks and the
-1 dBTP limiter takes ~2.3 dB off them, where Mixea let them clip over 0 dBTP. Exposing the
ceiling (-0.3 dBTP would give back ~0.7 dB) is the lever if the transients feel softer [H].
Only This Is Mine (Suno, -11.9 LUFS) through the same preset: -11.64 LUFS, quiet +2.0 dB,
loud 0.0 dB, side +1.2 dB, -1.0 dBTP, crest 14.6 -> 12.7 dB.
