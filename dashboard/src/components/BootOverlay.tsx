import { useEffect, useState } from "react";
import { fmtDuration } from "../format";
import type { LauncherStatus } from "../setup";

interface BootOverlayProps {
  launcher: LauncherStatus | null;
  liveStatus: string | null;
  livePhase: string | null;
  onBack: () => void;
}

type StepState = "done" | "active" | "todo" | "fail";

export default function BootOverlay({ launcher, liveStatus, livePhase, onBack }: BootOverlayProps) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const failed = launcher?.state === "failed" || liveStatus === "failed";
  const engine = launcher?.engine ?? "llm";
  const started = launcher?.state && launcher.state !== "idle" && launcher.state !== "launching";
  const published = liveStatus != null;
  const log = launcher?.log ?? [];
  const built = log.reduce((at, line, i) => (line.includes(":: building engine ::") ? i : at), -1);
  const onlineAt = log.reduce((at, line, i) => (line.includes(":: engine online in") ? i : at), -1);
  const online = onlineAt >= 0 && onlineAt > built;
  const loading = (livePhase === "loading_engine" || liveStatus === "loading_engine") && !online;
  const running = liveStatus === "running" && livePhase === "running";

  const steps: { label: string; state: StepState; detail?: string }[] = [
    {
      label: "Archive the previous run",
      state: started || published ? "done" : "active",
      detail: launcher?.archivedTo ? launcher.archivedTo.split(/[\\/]/).slice(-2).join("/") : "nothing to archive",
    },
    { label: "Start experiment.py", state: published ? "done" : started ? "active" : "todo" },
    ...(engine === "llm"
      ? [{ label: "Load Isolated PyTorch Tensors into VRAM", state: (running || online ? "done" : published ? "active" : "todo") as StepState, detail: "about a second per town" }]
      : []),
    { label: "Night 1: everyone decides", state: running ? "done" : published && (engine !== "llm" || !loading) ? "active" : "todo" },
  ];
  if (failed) {
    const active = steps.find((s) => s.state === "active");
    if (active) active.state = "fail";
  }
  const elapsed = launcher?.startedAt ? now / 1000 - launcher.startedAt : null;
  const tail = log.slice(-7);

  return (
    <section className="boot" role="status" aria-live="polite">
      <p className="eyebrow">{failed ? "THE TOWN DID NOT OPEN" : "OPENING THE TOWN"}</p>
      <h2>{failed ? "Startup failed" : engine === "llm" ? "Waking the minds" : "Rehearsing on instinct"}</h2>
      <ol className="boot__steps">
        {steps.map((step) => (
          <li key={step.label} data-state={step.state}>
            <i aria-hidden="true">{step.state === "done" ? "✓" : step.state === "fail" ? "✕" : step.state === "active" ? "◌" : "·"}</i>
            <span>{step.label}</span>
            {step.detail && <em>{step.detail}</em>}
          </li>
        ))}
      </ol>
      {launcher?.error && <p className="boot__error">{launcher.error}</p>}
      {tail.length > 0 && (
        <pre className="boot__log" aria-label="Engine log">
          {tail.join("\n")}
        </pre>
      )}
      <footer className="boot__foot">
        <span>{elapsed != null ? `${fmtDuration(elapsed)} elapsed` : ""}</span>
        {failed && (
          <button type="button" className="btn" onClick={onBack}>
            BACK TO SETUP
          </button>
        )}
      </footer>
    </section>
  );
}
