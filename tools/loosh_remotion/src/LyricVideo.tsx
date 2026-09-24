import React, {useEffect, useMemo, useState} from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  Sequence,
  continueRender,
  delayRender,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {useWindowedAudioData, visualizeAudio, visualizeAudioWaveform} from '@remotion/media-utils';

export type Word = {w: string; start: number; end: number};
export type Line = {start: number; end: number; text: string; size: 'big' | 'std'; words: Word[]};
export type Section = {start: number; end: number; section: string; still: string; move: string};
export type Song = {
  fps: number;
  duration: number;
  audio: string;
  sections: Section[];
  lines: Line[];
  drops: number[][];
  title: string;
  artist: string;
  album: string;
  motto: string;
  short: {start: number; cut: number; end: number; stills: string[]};
};

const CREAM = '#F1E4C2';
const EMBER = '#E85620';
const DARK = '#2B1A0E';
const FADE_S = 0.48;
const ZMAX = 1.12;

// ---------- font ----------
let fontPromise: Promise<void> | null = null;
const useRye = () => {
  const [handle] = useState(() => delayRender('load Rye'));
  useEffect(() => {
    if (!fontPromise) {
      const f = new FontFace('Rye', `url(${staticFile('fonts/Rye-Regular.ttf')})`);
      fontPromise = f.load().then((loaded) => {
        (document.fonts as FontFaceSet).add(loaded);
      });
    }
    fontPromise.then(() => continueRender(handle));
  }, [handle]);
};

const textShadow = '4px 4px 0 rgba(0,0,0,0.75), 0 0 2px rgba(0,0,0,0.9)';
const stroke = (px: number) => ({WebkitTextStroke: `${px}px ${DARK}`, paintOrder: 'stroke fill'} as React.CSSProperties);

// ---------- bass level (0..1) from the low bins, for the reactive bits ----------
const useBass = (audio: string) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const {audioData, dataOffsetInSeconds} = useWindowedAudioData({
    src: staticFile(audio),
    frame,
    fps,
    windowInSeconds: 10,
  });
  if (!audioData) return {bass: 0, wave: [] as number[]};
  const spec = visualizeAudio({audioData, frame, fps, numberOfSamples: 64, dataOffsetInSeconds, optimizeFor: 'speed'});
  const bass = Math.min(1, (spec[0] + spec[1] + spec[2]) / 3 * 1.6);
  const wave = visualizeAudioWaveform({audioData, frame, fps, numberOfSamples: 256, windowInSeconds: 1 / 12.5, dataOffsetInSeconds});
  return {bass, wave};
};

// ---------- one still with a Ken Burns move ----------
const Still: React.FC<{src: string; move: string; durationS: number; fadeIn: boolean; pulse: number}> = ({src, move, durationS, fadeIn, pulse}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const n = Math.max(1, Math.round(durationS * fps) - 1);
  const p = Math.min(1, frame / n);
  let scale = 1, tx = 0;
  if (move === 'push') scale = 1 + (ZMAX - 1) * p;
  else if (move === 'pull') scale = ZMAX - (ZMAX - 1) * p;
  else if (move === 'panR') {scale = ZMAX; tx = -(ZMAX - 1) * 50 * (2 * p - 1);}
  else if (move === 'panL') {scale = ZMAX; tx = (ZMAX - 1) * 50 * (2 * p - 1);}
  scale *= 1 + pulse * 0.035;
  const opacity = fadeIn ? interpolate(frame, [0, FADE_S * fps], [0, 1], {extrapolateRight: 'clamp'}) : 1;
  return (
    <AbsoluteFill style={{opacity}}>
      <Img src={staticFile(`stills/${src}`)} style={{width: '100%', height: '100%', objectFit: 'cover', transform: `scale(${scale}) translateX(${tx}%)`}} />
    </AbsoluteFill>
  );
};

// ---------- karaoke caption for the current line ----------
const Caption: React.FC<{lines: Line[]; t: number; vertical: boolean; bass: number}> = ({lines, t, vertical, bass}) => {
  const line = lines.find((l) => t >= l.start && t <= l.end + 0.6);
  if (!line) return null;
  const big = line.size === 'big';
  const fontSize = vertical ? (big ? 62 : 50) : big ? 66 : 48;
  const bottom = vertical ? 520 : 150;
  const kick = big ? 1 + bass * 0.04 : 1;
  return (
    <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', paddingBottom: bottom, paddingLeft: 80, paddingRight: 80}}>
      <div style={{fontFamily: 'Rye', fontSize, textAlign: 'center', lineHeight: 1.2, transform: `scale(${kick})`, textShadow, ...stroke(3)}}>
        {(() => {
          let activeIdx = -1;
          line.words.forEach((w, i) => {if (t >= w.start) activeIdx = i;});
          return line.words.map((w, i) => {
            const sung = i <= activeIdx;
            const active = i === activeIdx && t <= w.end + 0.15;
            return (
              <React.Fragment key={i}>
                <span style={{color: active ? EMBER : CREAM, opacity: sung ? 1 : 0.5, display: 'inline-block', transform: active ? 'scale(1.08)' : 'none'}}>{w.w}</span>
                {' '}
              </React.Fragment>
            );
          });
        })()}
      </div>
    </AbsoluteFill>
  );
};

// ---------- waveform strip, only inside the drop windows ----------
const Wave: React.FC<{wave: number[]; width: number; y: number; height: number}> = ({wave, width, y, height}) => {
  if (!wave.length) return null;
  const mid = y + height / 2;
  const pts = wave.map((v, i) => `${(i / (wave.length - 1)) * width},${mid - v * height * 1.4}`).join(' ');
  return (
    <svg width={width} height={y + height} style={{position: 'absolute', left: 0, top: 0}}>
      <polyline points={pts} fill="none" stroke={CREAM} strokeOpacity={0.85} strokeWidth={2} />
    </svg>
  );
};

const Bug: React.FC<{text: string}> = ({text}) => (
  <div style={{position: 'absolute', top: 40, right: 48, fontFamily: 'Rye', fontSize: 30, color: CREAM, opacity: 0.7, textShadow: '2px 2px 0 rgba(0,0,0,0.6)'}}>{text}</div>
);

const Card: React.FC<{lines: [string, number][]; opacity?: number}> = ({lines, opacity = 1}) => (
  <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity}}>
    <div style={{textAlign: 'center', fontFamily: 'Rye', color: CREAM, textShadow, ...stroke(3)}}>
      {lines.map(([s, size], i) => (
        <div key={i} style={{fontSize: size, lineHeight: 1.25}}>{s}</div>
      ))}
    </div>
  </AbsoluteFill>
);

const inDrop = (song: Song, t: number) => song.drops.some(([a, b]) => t >= a && t <= b);

// ================= landscape =================
export const LyricVideo: React.FC<{song: Song}> = ({song}) => {
  useRye();
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t = frame / fps;
  const {bass, wave} = useBass(song.audio);
  const drop = inDrop(song, t);
  const pulse = drop ? bass : 0;
  const endCardAt = song.duration - 6;
  const introEnd = song.sections[0].end;
  const endOpacity = interpolate(t, [endCardAt, endCardAt + 0.6], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      <Audio src={staticFile(song.audio)} />
      {song.sections.map((s, i) => {
        const last = i === song.sections.length - 1;
        const dur = s.end - s.start + (last ? 0 : FADE_S);
        return (
          <Sequence key={i} from={Math.round(s.start * fps)} durationInFrames={Math.round(dur * fps)} layout="none">
            <Still src={s.still} move={s.move} durationS={dur} fadeIn={i > 0} pulse={pulse} />
          </Sequence>
        );
      })}
      {drop && t < endCardAt ? <Wave wave={wave} width={width} y={height - 140} height={110} /> : null}
      {t < endCardAt ? <Caption lines={song.lines} t={t} vertical={false} bass={bass} /> : null}
      {t > introEnd - 1 && t < endCardAt ? <Bug text={song.artist} /> : null}
      {t >= endCardAt ? <Card lines={[[song.artist, 96], [song.album, 48], [song.motto, 34]]} opacity={endOpacity} /> : null}
    </AbsoluteFill>
  );
};

// ================= vertical short =================
export const ShortVideo: React.FC<{song: Song}> = ({song}) => {
  useRye();
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t0 = song.short.start;
  const t = t0 + frame / fps;
  const {bass, wave} = useBass(song.audio);
  const cutF = Math.round((song.short.cut - t0) * fps);
  const total = Math.round((song.short.end - t0) * fps);
  const drop = t >= song.short.cut;
  const pulse = drop ? bass : 0;
  const fadeOut = interpolate(frame, [total - 12, total], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const endTag = t >= song.short.end - 4.5;
  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      <Audio src={staticFile(song.audio)} startFrom={Math.round(t0 * fps)} volume={(f) => interpolate(f, [total - 38, total], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})} />
      <Sequence from={0} durationInFrames={cutF + Math.round(FADE_S * fps)} layout="none">
        <Still src={song.short.stills[0]} move="pull" durationS={(cutF + FADE_S * fps) / fps} fadeIn={false} pulse={0} />
      </Sequence>
      <Sequence from={cutF} durationInFrames={total - cutF} layout="none">
        <Still src={song.short.stills[1]} move="push" durationS={(total - cutF) / fps} fadeIn pulse={pulse} />
      </Sequence>
      {drop ? <Wave wave={wave} width={width} y={height - 460} height={90} /> : null}
      {!endTag ? <Caption lines={song.lines} t={t} vertical bass={bass} /> : null}
      {frame < 5.5 * fps ? <Card lines={[[song.artist, 54], [song.title, 30]]} opacity={interpolate(frame, [4.5 * fps, 5.5 * fps], [1, 0], {extrapolateLeft: 'clamp'})} /> : null}
      {frame >= 5.5 * fps ? <div style={{position: 'absolute', top: 100, width: '100%', textAlign: 'center', fontFamily: 'Rye', fontSize: 34, color: CREAM, opacity: 0.7}}>{song.artist}</div> : null}
      {endTag ? (
        <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 520}}>
          <div style={{fontFamily: 'Rye', fontSize: 44, color: CREAM, textShadow, ...stroke(3)}}>Full song on the channel</div>
        </AbsoluteFill>
      ) : null}
      <AbsoluteFill style={{backgroundColor: '#000', opacity: 1 - fadeOut}} />
    </AbsoluteFill>
  );
};
