#!/bin/bash
# Build an ident audio bed from Mixkit stock files: a hit aligned so its transient lands
# at HIT_T (frame 12 at 24 fps), a sub rumble + riser under the black lead-in, quiet
# embers under the hold, a 150 ms fade ending at the hard cut (2.67 s). Plain ASCII.
# usage: build_ident_mix.sh <label> <hit file> <hit transient time s> [hit gain dB]
set -e
LABEL="$1"; HIT="$2"; PEAK="$3"; GAIN="${4:-0}"
SRC=/Volumes/SSD1/Code/MusicGen/.mvwork/ident/sfx_research/mixkit
OUT=/Volumes/SSD1/Code/MusicGen/tools/loosh_remotion/public/audio/ident_mix_${LABEL}.wav
HIT_T=0.5; CUT=2.67; TOTAL=3.0
# how to place the hit: trim its front if the transient is later than HIT_T, else delay it
if awk "BEGIN{exit !($PEAK > $HIT_T)}"; then
  HIT_IN="-ss $(awk "BEGIN{print $PEAK-$HIT_T}") -i $HIT"; HIT_DELAY=0
else
  HIT_IN="-i $HIT"; HIT_DELAY=$(awk "BEGIN{printf \"%d\", ($HIT_T-$PEAK)*1000}")
fi
RISER_TRIM=$(awk "BEGIN{print 2.535-$HIT_T}")   # mixkit 790 peaks at its end (2.535 s): end it on the hit
ffmpeg -y -loglevel error \
  $HIT_IN \
  -ss 6 -t 0.6 -i $SRC/mixkit_1177.wav \
  -ss $RISER_TRIM -i $SRC/mixkit_790.wav \
  -ss 3 -t 2.4 -i $SRC/mixkit_1330.wav \
  -filter_complex "
    [0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=${GAIN}dB,adelay=${HIT_DELAY}|${HIT_DELAY}[hit];
    [1:a]aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:st=0:d=0.35,afade=t=out:st=0.45:d=0.15,volume=-6dB[rumble];
    [2:a]aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:st=0:d=0.2,volume=-4dB[riser];
    [3:a]aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:st=0:d=0.3,volume=-20dB,adelay=600|600[embers];
    [rumble][riser][hit][embers]amix=inputs=4:normalize=0:duration=longest,
    afade=t=out:st=$(awk "BEGIN{print $CUT-0.15}"):d=0.15,atrim=0:$TOTAL,apad=whole_dur=$TOTAL,
    alimiter=limit=0.89:level=false[out]" \
  -map "[out]" -ar 48000 -ac 2 "$OUT"
echo "$OUT"
