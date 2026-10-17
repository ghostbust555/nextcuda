/** The only module that talks to the server. */
import type { Answer, Gpu, Info, Made, Progress } from "./types.js";

/** A server path made relative to the page, so the page works on its server's own port and through the front door
 *  (`/open/<id>/`) alike; a link the server made absolute (from the Host it was asked on) keeps only its path. */
export function here(u: string): string {
  if (/^(data|blob):/.test(u)) return u;
  const p = /^https?:\/\//.test(u) ? new URL(u).pathname : u;
  return p.startsWith("/") ? p.slice(1) : p;
}

const linked = (m: Made): Made => ({ ...m, url: here(m.url), picture: m.picture ? here(m.picture) : m.picture });

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

async function post(path: string, body: Record<string, unknown>): Promise<Answer> {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = (await r.json()) as Answer & { error?: { message: string } };
  if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
  if (Array.isArray(j.data)) j.data = j.data.map(linked);
  return j;
}

export const api = {
  info: () => getJSON<Info>("api/info"),
  progress: () => getJSON<Progress>("api/progress"),
  gpu: () => getJSON<Gpu>("api/gpu"),
  history: async () => (await getJSON<{ data: Made[] }>("api/history")).data.map(linked),
  cancel: async () => { await fetch("api/cancel", { method: "POST" }); },
  /** a mesh from pictures; its answer links the GLB (an error carries the server's message) */
  generate: (body: Record<string, unknown>) => post("v1/model3d/generate", body),
  /** a mesh rigged or animated */
  rig: (body: Record<string, unknown>) => post("v1/model3d/rig", body),
};
