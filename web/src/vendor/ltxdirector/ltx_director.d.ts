// Types for the vendored (GPL-3) LTXDirector timeline editor. Only the symbols we mount.
export declare class TimelineEditor {
  constructor(node: unknown, container: HTMLElement, domWidget: unknown);
  destroy?(): void;
  timelineDataWidget?: { value?: string };
  // add an image segment from File objects at an optional frame position (used to inject generated stills)
  handleImageUpload(files: File[] | FileList, frameStart?: number, gapLength?: number): void;
}
export declare function parseInitial(jsonStr?: string): unknown;
