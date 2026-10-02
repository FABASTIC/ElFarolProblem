import { defineConfig, type Connect, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { simLauncher } from "./sim-launcher";

const here = path.dirname(fileURLToPath(import.meta.url));

const MIME: Record<string, string> = {
  ".json": "application/json; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".png": "image/png",
  ".pdf": "application/pdf",
  ".svg": "image/svg+xml",
};

function experimentData(root: string): Plugin {
  const base = path.resolve(root);
  const handler: Connect.NextHandleFunction = (req, res) => {
    if (req.method !== "GET" && req.method !== "HEAD") {
      res.statusCode = 405;
      res.end();
      return;
    }
    const [rawPath, query = ""] = (req.url ?? "/").split("?");
    const optional = new URLSearchParams(query).has("optional");
    let relative: string;
    try {
      relative = decodeURIComponent(rawPath);
    } catch {
      res.statusCode = 400;
      res.end();
      return;
    }
    const target = path.resolve(base, `.${relative}`);
    if (target !== base && !target.startsWith(base + path.sep)) {
      res.statusCode = 403;
      res.end();
      return;
    }
    fs.stat(target, (error, stat) => {
      res.setHeader("Cache-Control", "no-store");
      if (error || !stat.isFile()) {
        res.statusCode = optional ? 204 : 404;
        res.end();
        return;
      }
      res.statusCode = 200;
      res.setHeader("Content-Type", MIME[path.extname(target).toLowerCase()] ?? "application/octet-stream");
      res.setHeader("Content-Length", String(stat.size));
      if (req.method === "HEAD") {
        res.end();
        return;
      }
      fs.createReadStream(target)
        .on("error", () => {
          res.statusCode = 500;
          res.end();
        })
        .pipe(res);
    });
  };
  return {
    name: "elfarol-experiment-data",
    configureServer(server) {
      server.middlewares.use("/data", handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/data", handler);
    },
  };
}

export default defineConfig(() => {
  const dataDir = process.env.ELFAROL_DATA_DIR
    ? path.resolve(process.env.ELFAROL_DATA_DIR)
    : path.resolve(here, "../outputs/experiment");
  return {
    plugins: [react(), experimentData(dataDir), simLauncher({ repoRoot: path.resolve(here, ".."), dataDir })],
    server: { port: 5173 },
    preview: { port: 4173 },
    build: { target: "es2022", chunkSizeWarningLimit: 1800 },
  };
});
