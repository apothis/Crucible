// Types for the vendored (GPL-3) LTXDirector timeline editor. Only the symbols we mount.
export declare class TimelineEditor {
  constructor(node: unknown, container: HTMLElement, domWidget: unknown);
  destroy?(): void;
  timelineDataWidget?: { value?: string };
}
export declare function parseInitial(jsonStr?: string): unknown;
