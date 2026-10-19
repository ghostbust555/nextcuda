/** /api/state and /api/spec (glue/serve/src/home). */

export interface Proc {
  pid: number;
  vram_mb: number;
  program: string;
  kind?: string;
  command?: string;
  model?: string | null;
  /** its model's registry id, when its file is a registered one */
  model_id?: string | null;
  port?: string | null;
  gpus?: string[];
  args?: string;
}

/** a model vouched for on a card: `mb` all of it, `headroom_mb` of that its working memory beyond the weights (KV for
 *  its context, arenas, a song's or an image's activations) - from its model card when given, else from its files */
export interface Vouched { model: string; kind: string; mb: number; headroom_mb: number; headroom_given: boolean; measured: boolean; state: string; of?: number }

export interface Card {
  index: number;
  name: string;
  pci: string;
  vram_used_mb: number;
  vram_total_mb: number;
  busy_pct: number;
  power_w: number | null;
  power_cap_w: number | null;
  temp_pkg: number | null;
  temp_vram: number | null;
  pcie: { cur: { gen: number | null; width: number } } | null;
  procs: Proc[];
  vouched: Vouched[];
  vouched_mb: number;
  vouched_headroom_mb: number;
  overbooked: boolean;
}

export type ModelState = "disabled" | "unloaded" | "loading" | "loaded" | "busy" | "ready";

export interface Model {
  id: string;
  kind: "llm" | "image" | "audio" | "video";
  title: string | null;
  arch: string | null;
  enabled: boolean;
  gpus: number[];
  state: ModelState;
  vouch_mb: Record<string, number>;
  /** of vouch_mb, the headroom (Vouched) */
  headroom_mb: Record<string, number>;
  headroom_given: boolean;
  measured: boolean;
  page: number;
}

export interface Service { id: string; title: string; what: string; port: number; up: boolean; detail?: string | null; link?: string; href?: string | null; api?: string }

export interface Run { id: number; label: string; started: number; rc: number | null; out: string }

export interface State {
  ts: number;
  host: { mem_total_mb: number; mem_avail_mb: number; load1: number | null; cpus: number };
  cards: Card[];
  models: Model[];
  services: Service[];
  runs: Run[];
  idle_minutes: number;
  ports: { video: number; chat_ui: number };
  /** maintenance mode while it is on (system.maintenance): the notice callers read, the bypass key */
  maintenance: Maintenance | null;
  /** the models a card offers that are not downloaded (system.models' downloaded: false) */
  available: Available[];
  /** the downloads under way and the last ones, newest first (system.models.pulls) */
  pulls: Pull[];
  /** NS_AUTO_PULL: a request for a model not downloaded fetches it and waits */
  auto_pull: boolean;
}

export interface Available {
  id: string; kind: string; title?: string; about?: string; bytes: number;
  /** the drive and folder it would go to, whether it has the room, its free bytes */
  drive: string; folder: string; fits: boolean; free: number;
  pull: Pull | null;
  /** whether this box's cards can run it (its memory against theirs), and why not */
  runs_here?: boolean; why_not?: string | null;
}

export interface Pull {
  model: string; kind: string; title: string; drive: string; folder: string;
  state: "queued" | "downloading" | "verifying" | "done" | "failed" | "cancelled";
  error: string | null; bytes_total: number; bytes_done: number; pct: number; speed: number; eta_s: number | null;
  started: number; ended: number | null; run: number;
}

export interface Maintenance { on: boolean; message: string; since: number; until: number | null; retry_after_s: number; notice: string; bypass?: string }

export interface Operation {
  tags: string[];
  operationId?: string;
  summary: string;
  requestBody?: { content: { "application/json": { example?: unknown; schema?: { properties?: Record<string, { type: string; description?: string; enum?: string[] }>; required?: string[] } } } };
  parameters?: unknown[];
  "x-curl"?: string;
}

export interface Spec {
  token: string;
  spec: { info: { description: string; version: string }; servers: { url: string }[]; paths: Record<string, Record<string, Operation>> };
}

export interface SettingRow {
  name: string;
  group?: string;
  default?: string;
  what: string;
  readers: string[];
  file: string | null;
  env?: string | null;
}

export interface Config {
  path: string;
  text: string;
  settings: SettingRow[];
  other: SettingRow[];
  prefixes: { prefix: string; what: string; readers: string[] }[];
}

export interface ConfigSaved { changed: string[]; restart: string[]; note?: string | null }
