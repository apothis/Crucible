import React, {useEffect, useState} from 'react';
import {
  AbsoluteFill,
  Audio,
  continueRender,
  delayRender,
  interpolate,
  random,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

// Apotheon channel ident ("Ember strike"), 3.0 s at 24 fps = 72 frames.
// Same for every video apart from the song title. Brand = the "modern" look in
// backend/wordmark.py: Josefin Sans 300, tracked caps, flat ivory, ember accent hairline.
//
//   0.00-0.50  black, film grain, a low ember haze, a few embers drifting up, rumble
//   0.50       HIT: 2-frame flash, the wordmark slams in (scale 1.08 -> 1.0, chromatic
//              split collapsing, 4-frame shake), the accent hairline draws out
//   0.90-2.50  hold: one specular sheen pass over the letters, song title fades up,
//              embers keep rising and thin out, the word breathes (1.00 -> 1.015)
//   2.67       hard cut to black (no fade: the video butts straight on)

export type IdentProps = {
  title: string;
  sub: string;
  audio: string | null;
};

const IVORY = 'rgb(236,231,220)';
const ACCENT = 'rgb(220,120,60)';
const HIT = 12; // frame of the impact (0.5 s)
const CUT = 64; // hard cut to black (2.67 s)

let fontPromise: Promise<void> | null = null;
const useJosefin = () => {
  const [handle] = useState(() => delayRender('load Josefin Sans'));
  useEffect(() => {
    if (!fontPromise) {
      const f = new FontFace('Josefin', `url(${staticFile('fonts/JosefinSans.ttf')})`, {weight: '100 700'});
      fontPromise = f.load().then((loaded) => {
        (document.fonts as FontFaceSet).add(loaded);
      });
    }
    fontPromise.then(() => continueRender(handle));
  }, [handle]);
};

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

// ---------- embers ----------
type Ember = {x: number; y0: number; speed: number; sway: number; phase: number; size: number; born: number; life: number; warm: number};
const EMBERS: Ember[] = Array.from({length: 70}, (_, i) => ({
  x: 8 + random(`x${i}`) * 84,
  y0: 62 + random(`y${i}`) * 50,
  speed: 10 + random(`s${i}`) * 16,
  sway: 1.2 + random(`w${i}`) * 2.4,
  phase: random(`p${i}`) * Math.PI * 2,
  size: 2 + random(`z${i}`) * 4,
  born: Math.floor(random(`b${i}`) * 40) - 10,
  life: 34 + random(`l${i}`) * 30,
  warm: random(`c${i}`),
}));

const Embers: React.FC<{frame: number; fps: number; boost: number}> = ({frame, fps, boost}) => (
  <AbsoluteFill style={{overflow: 'hidden'}}>
    {EMBERS.map((e, i) => {
      const age = frame - e.born;
      if (age < 0 || age > e.life) return null;
      const t = age / fps;
      const y = e.y0 - e.speed * t * (1 + boost * 0.6);
      const x = e.x + Math.sin(e.phase + t * e.sway) * 1.6;
      const fade = interpolate(age, [0, 6, e.life - 10, e.life], [0, 1, 0.8, 0], clamp);
      const col = e.warm > 0.6 ? 'rgb(255,214,150)' : e.warm > 0.3 ? 'rgb(240,140,60)' : 'rgb(200,70,20)';
      return (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: `${x}%`,
            top: `${y}%`,
            width: e.size,
            height: e.size,
            borderRadius: '50%',
            background: col,
            opacity: fade * 0.9,
            boxShadow: `0 0 ${e.size * 2.5}px ${e.size * 0.8}px rgba(230,110,40,${0.35 * fade})`,
            filter: `blur(${e.size > 4.5 ? 0.8 : 0.3}px)`,
          }}
        />
      );
    })}
  </AbsoluteFill>
);

// ---------- film grain ----------
const Grain: React.FC<{frame: number; amount: number}> = ({frame, amount}) => (
  <AbsoluteFill style={{opacity: amount, mixBlendMode: 'overlay', pointerEvents: 'none'}}>
    <svg width="100%" height="100%">
      <filter id="grain">
        <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed={frame % 97} stitchTiles="stitch" />
        <feColorMatrix type="saturate" values="0" />
      </filter>
      <rect width="100%" height="100%" filter="url(#grain)" />
    </svg>
  </AbsoluteFill>
);

export const Ident: React.FC<IdentProps> = ({title, sub, audio}) => {
  useJosefin();
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  if (frame >= CUT) {
    return (
      <AbsoluteFill style={{background: '#000'}}>
        {audio ? <Audio src={staticFile(audio)} /> : null}
      </AbsoluteFill>
    );
  }

  const since = frame - HIT;
  // pre-hit: the haze low in frame brightens as the rumble builds
  const haze = interpolate(frame, [0, HIT], [0.12, 0.42], clamp);
  // hit: flash (2 frames), slam scale, chromatic split, shake
  const flash = since < 0 ? 0 : interpolate(since, [0, 1, 3], [0.85, 0.35, 0], clamp);
  const slam = interpolate(since, [0, 7], [1.08, 1.0], {...clamp, easing: (t) => 1 - Math.pow(1 - t, 3)});
  const breathe = interpolate(since, [7, CUT - HIT], [1.0, 1.015], clamp);
  const split = interpolate(since, [0, 8], [7, 0], {...clamp, easing: (t) => 1 - Math.pow(1 - t, 2)});
  const shakeAmp = interpolate(since, [0, 4], [6, 0], clamp);
  const sx = since >= 0 ? (random(`shx${frame}`) - 0.5) * 2 * shakeAmp : 0;
  const sy = since >= 0 ? (random(`shy${frame}`) - 0.5) * 2 * shakeAmp : 0;
  const wordOpacity = since >= 0 ? 1 : 0;
  // hairline draws out under the word; title and sub fade up after
  const line = interpolate(since, [2, 12], [0, 1], {...clamp, easing: (t) => 1 - Math.pow(1 - t, 3)});
  const titleIn = interpolate(since, [10, 22], [0, 1], clamp);
  const titleRise = interpolate(since, [10, 22], [10, 0], {...clamp, easing: (t) => 1 - Math.pow(1 - t, 2)});
  const subIn = interpolate(since, [18, 28], [0, 0.55], clamp);
  // one specular pass across the letters, 0.9 s to 1.6 s
  const sheen = interpolate(since, [10, 26], [-60, 160], clamp);
  // the glow behind the word is hot on the hit and settles
  const glow = interpolate(since, [0, 2, 14], [0.0, 0.9, 0.35], clamp);
  const emberBoost = interpolate(since, [0, 2, 12], [0, 1, 0], clamp);

  const wordStyle: React.CSSProperties = {
    fontFamily: 'Josefin',
    fontWeight: 300,
    fontSize: 118,
    letterSpacing: '0.60em',
    textIndent: '0.60em', // balance the trailing tracking so the word is optically centred
    lineHeight: 1,
    textTransform: 'uppercase',
    whiteSpace: 'nowrap',
    position: 'absolute',
    left: 0,
    right: 0,
    textAlign: 'center',
  };

  return (
    <AbsoluteFill style={{background: '#050403', overflow: 'hidden'}}>
      {audio ? <Audio src={staticFile(audio)} /> : null}
      {/* ember haze, low in frame */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse 70% 45% at 50% 100%, rgba(220,90,30,${haze}) 0%, rgba(120,35,10,${haze * 0.5}) 40%, rgba(0,0,0,0) 75%)`,
        }}
      />
      <Embers frame={frame} fps={fps} boost={emberBoost} />

      {/* the stage: everything that shakes and breathes */}
      <AbsoluteFill style={{transform: `translate(${sx}px, ${sy}px) scale(${slam * breathe})`, opacity: wordOpacity}}>
        {/* hot glow behind the word */}
        <AbsoluteFill
          style={{
            background: `radial-gradient(ellipse 38% 18% at 50% 47%, rgba(230,120,50,${glow * 0.55}) 0%, rgba(0,0,0,0) 100%)`,
          }}
        />
        {/* chromatic split copies */}
        <div style={{...wordStyle, top: 455, color: 'rgba(255,40,40,0.55)', transform: `translateX(${-split}px)`, mixBlendMode: 'screen'}}>Apotheon</div>
        <div style={{...wordStyle, top: 455, color: 'rgba(40,200,255,0.55)', transform: `translateX(${split}px)`, mixBlendMode: 'screen'}}>Apotheon</div>
        {/* the wordmark */}
        <div style={{...wordStyle, top: 455, color: IVORY, textShadow: '0 0 28px rgba(0,0,0,0.9)'}}>Apotheon</div>
        {/* specular sheen clipped to the letters */}
        <div
          style={{
            ...wordStyle,
            top: 455,
            color: 'transparent',
            backgroundImage: `linear-gradient(100deg, rgba(255,255,255,0) 44%, rgba(255,236,210,0.95) 50%, rgba(255,255,255,0) 56%)`,
            backgroundSize: '300% 100%',
            backgroundPosition: `${sheen}% 0`,
            WebkitBackgroundClip: 'text',
            backgroundClip: 'text',
          }}
        >
          Apotheon
        </div>

        {/* ember accent hairline */}
        <div
          style={{
            position: 'absolute',
            top: 612,
            left: '50%',
            width: 220 * line,
            height: 2,
            transform: 'translateX(-50%)',
            background: ACCENT,
            boxShadow: `0 0 12px rgba(220,120,60,${0.6 * line})`,
          }}
        />

        {/* song title: light tracked caps, the modern look's title lettering */}
        <div
          style={{
            position: 'absolute',
            top: 646 + titleRise,
            left: 0,
            right: 0,
            textAlign: 'center',
            fontFamily: 'Josefin',
            fontWeight: 200,
            fontSize: 40,
            letterSpacing: '0.18em',
            textIndent: '0.18em',
            textTransform: 'uppercase',
            color: IVORY,
            opacity: titleIn * 0.92,
            whiteSpace: 'nowrap',
          }}
        >
          {title}
        </div>
        <div
          style={{
            position: 'absolute',
            top: 708,
            left: 0,
            right: 0,
            textAlign: 'center',
            fontFamily: 'Josefin',
            fontWeight: 300,
            fontSize: 17,
            letterSpacing: '0.50em',
            textIndent: '0.50em',
            textTransform: 'uppercase',
            color: 'rgb(150,145,136)',
            opacity: subIn,
          }}
        >
          {sub}
        </div>
      </AbsoluteFill>

      {/* vignette, grain, flash */}
      <AbsoluteFill style={{background: 'radial-gradient(ellipse 75% 70% at 50% 50%, rgba(0,0,0,0) 55%, rgba(0,0,0,0.75) 100%)'}} />
      <Grain frame={frame} amount={0.16} />
      <AbsoluteFill style={{background: 'rgb(255,240,220)', opacity: flash}} />
    </AbsoluteFill>
  );
};
