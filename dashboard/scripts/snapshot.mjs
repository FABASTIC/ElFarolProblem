import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = path.resolve(process.env.ELFAROL_DATA_DIR ?? path.join(here, "..", "..", "outputs", "experiment"));
const target = path.resolve(here, "..", "public", "data");
const force = process.argv.includes("--force");
const RUNNING = new Set(["booting", "loading_engine", "running", "analyzing"]);
const TRIAL_FILES = ["agent_epoch_details.csv", "mind_trace.csv", "minds.json"];

function fail(message) {
  console.error(`snapshot: ${message}`);
  process.exit(1);
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

const comparison = readJson(path.join(source, "comparison.json"));
if (!Array.isArray(comparison) || comparison.length === 0) fail(`no finished sweep in ${source} (comparison.json missing or empty)`);

const live = readJson(path.join(source, "live_state.json"));
if (live && RUNNING.has(live.status) && !force) fail(`a sweep is still ${live.status} in ${source}; wait for it to finish or pass --force`);

fs.rmSync(target, { recursive: true, force: true });
fs.mkdirSync(target, { recursive: true });

const ABSOLUTE = /^([A-Za-z]:[\\/]|\/mnt\/|\/home\/|\/Users\/|\/root\/)/;

function scrub(value) {
  if (typeof value === "string" && ABSOLUTE.test(value)) {
    const normalized = value.replace(/\\/g, "/");
    const at = normalized.indexOf("outputs/");
    return at >= 0 ? normalized.slice(at) : path.posix.basename(normalized);
  }
  if (Array.isArray(value)) return value.map(scrub);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, inner]) => [key, scrub(inner)]));
  return value;
}

const copied = [];
const copy = (relative) => {
  const from = path.join(source, relative);
  if (!fs.existsSync(from)) return false;
  const to = path.join(target, relative);
  fs.mkdirSync(path.dirname(to), { recursive: true });
  const parsed = relative.endsWith(".json") ? readJson(from) : null;
  if (parsed != null) fs.writeFileSync(to, JSON.stringify(scrub(parsed)));
  else fs.copyFileSync(from, to);
  copied.push([relative, fs.statSync(to).size]);
  return true;
};

copy("comparison.json");
copy("analytics_report.json");

const trials = [...new Set(comparison.map((row) => row.trial_dir).filter((dir) => typeof dir === "string" && /^[\w.-]+$/.test(dir)))];
for (const dir of trials) {
  for (const file of TRIAL_FILES) copy(path.join(dir, file));
}

const bytes = copied.reduce((sum, [, size]) => sum + size, 0);
console.log(`snapshot: ${trials.length} trials, ${copied.length} files, ${(bytes / 1048576).toFixed(1)} MiB -> ${path.relative(process.cwd(), target) || target}`);
