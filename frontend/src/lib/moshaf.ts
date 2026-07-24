import type { MoshafConfig, MoshafField } from "./types";

// The reciter's moshaf attributes: which madd lengths they hold, which rules they read.
// The schema is the backend's — introspected from MoshafAttributes — so this file never
// hard-codes the fields; it only fetches them and remembers the reciter's choices.

const KEY = "tajwid.moshaf";

let schema: Promise<MoshafField[]> | null = null;

/** The attribute schema (fields, options, defaults), fetched once from the backend. */
export function loadMoshafSchema(): Promise<MoshafField[]> {
  if (!schema) {
    schema = fetch("/api/moshaf-schema")
      .then((r) => {
        if (!r.ok) throw new Error(`moshaf-schema: ${r.status}`);
        return r.json();
      })
      .then((d) => d.fields as MoshafField[])
      .catch((e) => {
        schema = null; // let a later open retry
        throw e;
      });
  }
  return schema;
}

/** Every attribute at its starting value — a COMPLETE config, so required fields hold. */
export function defaultConfig(fields: MoshafField[]): MoshafConfig {
  return Object.fromEntries(fields.map((f) => [f.key, f.default]));
}

export function loadMoshafConfig(): MoshafConfig | null {
  try {
    const s = localStorage.getItem(KEY);
    return s ? (JSON.parse(s) as MoshafConfig) : null;
  } catch {
    return null;
  }
}

export function saveMoshafConfig(cfg: MoshafConfig | null): void {
  try {
    if (cfg) localStorage.setItem(KEY, JSON.stringify(cfg));
    else localStorage.removeItem(KEY);
  } catch {
    /* private mode / disabled storage — the setting just won't persist */
  }
}
