import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = path.resolve(here, "..", "..", "analyzer.py");
const target = path.resolve(here, "..", "public", "engine", "analyzer.py");

if (fs.existsSync(source)) {
  const text = fs.readFileSync(source, "utf8");
  const current = fs.existsSync(target) ? fs.readFileSync(target, "utf8") : null;
  if (current !== text) {
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, text);
    console.log("synced analyzer.py into public/engine");
  }
} else if (!fs.existsSync(target)) {
  console.error("analyzer.py not found next to the dashboard and no bundled copy exists");
  process.exit(1);
}
