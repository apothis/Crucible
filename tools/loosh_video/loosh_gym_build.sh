#!/bin/bash
# Photo-story lyric video build for Loosh Woggins - She Left Me for a Gym Membership
# usage: ./loosh_gym_build.sh sections | assemble | short
set -e
cd /Volumes/SSD1/Code/MusicGen/.mvwork
WAV="/Volumes/SSD1/Downloads/Gym Membership (1).wav"
STILLS=loosh_gym_stills
B=loosh_gym_build
FONT=/Volumes/SSD1/Code/MusicGen/.mvwork/fonts/Rye-Regular.ttf
FPS=25
FADE=0.48            # 12 frames
ZMAX=1.12
mkdir -p "$B"

read_cues() { grep -v '^#' loosh_gym_cues.csv | tail -n +2; }

# ---------- pass 1: one clip per section ----------
sections() {
  i=0; n=$(read_cues | wc -l | tr -d ' ')
  read_cues | while IFS=, read -r s e sec still move src; do
    dur=$(echo "$e - $s" | bc -l)
    last=$(( i == n-1 ))
    if [ $last = 1 ]; then ext=0; else ext=$FADE; fi
    N=$(printf '%.0f' "$(echo "($dur + $ext) * $FPS" | bc -l)")
    case $move in
      push) Z="1+($ZMAX-1)*on/($N-1)"; X="iw/2-(iw/zoom/2)"; Y="ih/2-(ih/zoom/2)";;
      pull) Z="$ZMAX-($ZMAX-1)*on/($N-1)"; X="iw/2-(iw/zoom/2)"; Y="ih/2-(ih/zoom/2)";;
      panR) Z="$ZMAX"; X="(iw-iw/zoom)*on/($N-1)"; Y="ih/2-(ih/zoom/2)";;
      panL) Z="$ZMAX"; X="(iw-iw/zoom)*(1-on/($N-1))"; Y="ih/2-(ih/zoom/2)";;
      *)    Z="1"; X="0"; Y="0";;
    esac
    out=$(printf '%s/sec_%02d_%s.mp4' "$B" "$i" "$sec")
    echo "[$i] $sec $s-$e ($dur s, $N frames, $move) -> $out"
    ffmpeg -nostdin -v error -y -i "$STILLS/$still.webp" \
      -vf "crop=1536:864:0:80,scale=3840:2160:flags=lanczos,zoompan=z='$Z':x='$X':y='$Y':d=${N}:s=1920x1080:fps=$FPS,format=yuv420p" \
      -frames:v ${N} -c:v libx264 -preset fast -crf 14 -r $FPS "$out"
    i=$((i+1))
  done
}

# ---------- pass 2: crossfade, captions, waveform, audio ----------
assemble() {
  inputs=(); fc=""; i=0; offs=()
  read_cues | while IFS=, read -r s e sec still move src; do echo "$e"; done > "$B/ends.txt"
  ENDS=(); while read -r l; do ENDS+=("$l"); done < "$B/ends.txt"
  n=${#ENDS[@]}
  for f in $(ls "$B"/sec_*.mp4 | sort); do inputs+=(-i "$f"); done
  # xfade chain: offset = end of the previous section on the master timeline
  prev="[0:v]"
  for ((i=1;i<n;i++)); do
    off=${ENDS[$((i-1))]}
    lab="[v$i]"; [ $i = $((n-1)) ] && lab="[vx]"
    fc="$fc$prev[$i:v]xfade=transition=fade:duration=$FADE:offset=$off$lab;"
    prev="$lab"
  done
  # captions + branding: timed PNG overlays rendered by loosh_gym_overlays.py
  k=$n; ov=""; prev="[vx]"
  while IFS=, read -r png os oe; do
    inputs+=(-i "$png")
    ov="$ov$prev[$k:v]overlay=0:0:enable='between(t,$os,$oe)':format=auto[o$k];"
    prev="[o$k]"; k=$((k+1))
  done < "$B/overlays.txt"
  A=$k
  # waveform only on the drop windows
  WEN="between(t,58.5,71.7)+between(t,116.5,127.3)+between(t,163.5,180)+between(t,183.5,186.5)+between(t,189,195)"
  fc="$fc$ov"
  fc="$fc[$A:a]showwaves=s=1920x110:mode=cline:draw=full:colors=0xF1E4C2@0.85:rate=$FPS,format=rgba[wv];"
  fc="$fc$prev[wv]overlay=x=0:y=H-h-30:enable='$WEN':format=auto[vo]"
  ffmpeg -nostdin -v error -y "${inputs[@]}" -i "$WAV" -filter_complex "$fc" -map "[vo]" -map "$A:a" \
    -t 196.361 -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r $FPS -c:a aac -b:a 320k -movflags +faststart \
    "$B/Loosh_Woggins_-_She_Left_Me_for_a_Gym_Membership_lyric_video_1080p.mp4"
}

case "$1" in
  sections) sections;;
  assemble) assemble;;
  *) echo "usage: $0 sections|assemble"; exit 1;;
esac
