import React from 'react';
import {Composition} from 'remotion';
import {LyricVideo, ShortVideo, type Song} from './LyricVideo';
import bbbd from '../public/data/bbbd.json';

const song = bbbd as Song;

export const Root: React.FC = () => {
  const fps = song.fps;
  const shortLen = song.short.end - song.short.start;
  return (
    <>
      <Composition
        id="Landscape"
        component={LyricVideo}
        durationInFrames={Math.round(song.duration * fps)}
        calculateMetadata={({props}) => ({durationInFrames: Math.round(props.song.duration * props.song.fps)})}
        fps={fps}
        width={1920}
        height={1080}
        defaultProps={{song}}
      />
      <Composition
        id="Short"
        component={ShortVideo}
        durationInFrames={Math.round(shortLen * fps)}
        calculateMetadata={({props}) => ({durationInFrames: Math.round((props.song.short.end - props.song.short.start) * props.song.fps)})}
        fps={fps}
        width={1080}
        height={1920}
        defaultProps={{song}}
      />
    </>
  );
};
