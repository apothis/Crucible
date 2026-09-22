#!/bin/bash
# Vertical Short: chorus 1 + drop 1 of Gym Membership, 1080x1920, 39.0-71.7 s of the track
set -e
cd /Volumes/SSD1/Code/MusicGen/.mvwork
WAV="/Volumes/SSD1/Downloads/Gym Membership (1).wav"; S=loosh_gym_stills; B=loosh_gym_build/short
FPS=25; FADE=0.48; T0=39.0; CUT=58.5; T1=71.7
D1=$(echo "$CUT - $T0 + $FADE" | bc -l); D2=$(echo "$T1 - $CUT" | bc -l)
N1=$(printf '%.0f' "$(echo "$D1 * $FPS" | bc -l)"); N2=$(printf '%.0f' "$(echo "$D2 * $FPS" | bc -l)")
PRE="crop=864:1536:80:0,scale=2160:3840:flags=lanczos"
ffmpeg -nostdin -v error -y -i "$S/GYM-04p_on_my_knees_portrait.webp" -vf "$PRE,zoompan=z='1.12-0.12*on/(${N1}-1)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${N1}:s=1080x1920:fps=$FPS,format=yuv420p" -frames:v ${N1} -c:v libx264 -preset fast -crf 14 -r $FPS "$B/a.mp4"
ffmpeg -nostdin -v error -y -i "$S/GYM-05p_the_drop_portrait.webp" -vf "$PRE,zoompan=z='1+0.12*on/(${N2}-1)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${N2}:s=1080x1920:fps=$FPS,format=yuv420p" -frames:v ${N2} -c:v libx264 -preset fast -crf 14 -r $FPS "$B/b.mp4"
OFF=$(echo "$CUT - $T0" | bc -l); DUR=$(echo "$T1 - $T0" | bc -l)
inputs=(-i "$B/a.mp4" -i "$B/b.mp4"); fc="[0:v][1:v]xfade=transition=fade:duration=$FADE:offset=$OFF[vx];"
k=2; prev="[vx]"
while IFS=, read -r png os oe; do inputs+=(-i "$png"); fc="$fc$prev[$k:v]overlay=0:0:enable='between(t,$os,$oe)':format=auto[o$k];"; prev="[o$k]"; k=$((k+1)); done < "$B/overlays.txt"
A=$k
# waveform as a PNG sequence (the in-graph showwaves overlay mangles colour on this vertical pipeline)
rm -rf "$B/wv"; mkdir -p "$B/wv"
ffmpeg -nostdin -v error -y -ss $CUT -t $D2 -i "$WAV" -filter_complex "[0:a]showwaves=s=1080x90:mode=cline:draw=full:colors=0xF1E4C2@0.85:rate=$FPS,format=rgba[wv]" -map "[wv]" "$B/wv/%04d.png"
inputs+=(-framerate $FPS -itsoffset $OFF -i "$B/wv/%04d.png")
fc="$fc$prev[$A:v]overlay=x=0:y=1460:enable='gte(t,$OFF)':eof_action=pass:format=auto[vo];"
fc="$fc[$((A+1)):a]afade=t=out:st=$(echo "$DUR - 1.5" | bc -l):d=1.5[ao]"
ffmpeg -nostdin -v error -y "${inputs[@]}" -ss $T0 -t $DUR -i "$WAV" -filter_complex "$fc" -map "[vo]" -map "[ao]" -t $DUR \
  -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r $FPS -c:a aac -b:a 320k -movflags +faststart \
  "$B/Loosh_Woggins_-_Gym_Membership_SHORT_1080x1920.mp4"
