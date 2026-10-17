/** The only module that talks to the server. */
import type { GenAnswer, Gpu, Info, Picture, Progress } from "./types.js";

/** A server path made relative to the page, so the page works on its server's own port and through the front door
 *  (`/open/<id>/`) alike; a link the server made absolute (from the Host it was asked on) keeps only its path. */
export function here(u: string): string {
  if (/^(data|blob):/.test(u)) return u;
  const p = /^https?:\/\//.test(u) ? new URL(u).pathname : u;
  return p.startsWith("/") ? p.slice(1) : p;
}

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  info: () => getJSON<Info>("api/info"),
  progress: () => getJSON<Progress>("api/progress"),
  gpu: () => getJSON<Gpu>("api/gpu"),
  history: async () => (await getJSON<{ data: Picture[] }>("api/history")).data.map((p) => ({ ...p, url: here(p.url) })),

  /** OpenAI's images endpoint (an edit: the edits one, its pictures as data URLs), with our fields; an error carries
   *  the server's message. */
  async generate(body: Record<string, unknown>, edit = false): Promise<GenAnswer> {
    const r = await fetch(edit ? "v1/images/edits" : "v1/images/generations", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = (await r.json()) as GenAnswer & { error?: { message: string } };
    if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
    for (const d of j.data ?? []) if (d.url) d.url = here(d.url);
    return j;
  },
};
