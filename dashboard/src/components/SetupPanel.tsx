import { useEffect, useState, type KeyboardEvent } from "react";
import { fmtDuration } from "../format";
import type { Launcher } from "../launcher";
import {
  COMFORT_THRESHOLD,
  LIMITS,
  PRESETS,
  RECOMMENDED,
  adviseConfig,
  estimateRun,
  sanitizeConfig,
  type Engine,
  type LaunchConfig,
} from "../setup";

interface SetupPanelProps {
  draft: LaunchConfig;
  onDraft: (config: LaunchConfig) => void;
  launcher: Launcher;
  calibration: number | null;
  canClose: boolean;
  onClose: () => void;
  onBegin: (config: LaunchConfig) => void;
}

interface Mark {
  value: number;
  label: string;
  tone: "star" | "line";
}

interface DialProps {
  id: string;
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  sliderMax: number;
  marks: Mark[];
  hint: string;
  onChange: (value: number) => void;
}

function Dial({ id, label, unit, value, min, max, sliderMax, marks, hint, onChange }: DialProps) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const parsed = Number(text);
  const valid = text.trim() !== "" && Number.isInteger(parsed) && parsed >= min && parsed <= max;
  const commit = (next: number) => onChange(Math.max(min, Math.min(max, Math.round(next))));
  const pct = (v: number) => `${((Math.min(v, sliderMax) - min) / (sliderMax - min)) * 100}%`;

  return (
    <div className="dial">
      <div className="dial__head">
        <label htmlFor={id}>{label}</label>
        <span className="dial__hint">{hint}</span>
      </div>
      <div className="dial__row">
        <button type="button" className="btn dial__step" onClick={() => commit(value - 1)} disabled={value <= min} aria-label={`Decrease ${label}`}>
          −
        </button>
        <input
          id={id}
          className="dial__input"
          inputMode="numeric"
          value={text}
          aria-invalid={!valid}
          onChange={(event) => {
            setText(event.target.value);
            const next = Number(event.target.value);
            if (event.target.value.trim() !== "" && Number.isInteger(next) && next >= min && next <= max) onChange(next);
          }}
          onBlur={() => setText(String(value))}
          onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
            if (event.key === "ArrowUp") {
              event.preventDefault();
              commit(value + (event.shiftKey ? 10 : 1));
            } else if (event.key === "ArrowDown") {
              event.preventDefault();
              commit(value - (event.shiftKey ? 10 : 1));
            }
          }}
        />
        <span className="dial__unit">{unit}</span>
        <button type="button" className="btn dial__step" onClick={() => commit(value + 1)} disabled={value >= max} aria-label={`Increase ${label}`}>
          +
        </button>
      </div>
      <div className="dial__slider">
        <input
          type="range"
          min={min}
          max={sliderMax}
          step={1}
          value={Math.min(value, sliderMax)}
          onChange={(event) => onChange(Number(event.target.value))}
          aria-label={`${label} slider`}
          style={{ ["--fill" as string]: pct(value) }}
        />
        {marks.map((mark) => (
          <span key={`${mark.tone}${mark.value}`} className="dial__mark" data-tone={mark.tone} style={{ left: pct(mark.value) }} title={mark.label}>
            <i aria-hidden="true" />
            <em>{mark.label}</em>
          </span>
        ))}
      </div>
      {!valid && (
        <p className="dial__error" role="alert">
          Enter a whole number from {min} to {max}.
        </p>
      )}
    </div>
  );
}

function SeedEditor({ seeds, onChange }: { seeds: number[]; onChange: (seeds: number[]) => void }) {
  const [entry, setEntry] = useState("");
  const add = () => {
    const value = Number(entry);
    if (entry.trim() === "" || !Number.isInteger(value) || value < 0) return;
    if (!seeds.includes(value) && seeds.length < LIMITS.seeds.max) onChange([...seeds, value]);
    setEntry("");
  };
  return (
    <div className="seeds">
      {seeds.map((seed) => (
        <span key={seed} className="seeds__chip">
          {seed}
          <button
            type="button"
            aria-label={`Remove seed ${seed}`}
            onClick={() => onChange(seeds.filter((s) => s !== seed))}
            disabled={seeds.length <= LIMITS.seeds.min}
          >
            ×
          </button>
        </span>
      ))}
      {seeds.length < LIMITS.seeds.max && (
        <input
          className="seeds__add"
          inputMode="numeric"
          placeholder="+ seed"
          value={entry}
          aria-label="Add seed"
          onChange={(event) => setEntry(event.target.value.replace(/[^0-9]/g, ""))}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === ",") {
              event.preventDefault();
              add();
            }
          }}
          onBlur={add}
        />
      )}
    </div>
  );
}

const ENGINES: { id: Engine; title: string; sub: string; lock?: string }[] = [
  { id: "llm", title: "ISOLATED PYTORCH TENSORS", sub: "One brain per agent, batched on the GPU" },
  { id: "rehearsal", title: "REHEARSAL", sub: "CPU brains · needs Torch on Windows", lock: "GPU ONLY" },
];

export default function SetupPanel({ draft, onDraft, launcher, calibration, canClose, onClose, onBegin }: SetupPanelProps) {
  const [armed, setArmed] = useState(false);
  const { config, errors } = sanitizeConfig(draft);
  const signature = JSON.stringify(draft);
  useEffect(() => setArmed(false), [signature]);
  useEffect(() => {
    if (!armed) return;
    const id = window.setTimeout(() => setArmed(false), 6000);
    return () => window.clearTimeout(id);
  }, [armed]);
  const estimate = estimateRun(draft, calibration);
  const advice = adviseConfig(draft);
  const capability = launcher.status?.capabilities?.[draft.engine];
  const offline = launcher.reachable === false;
  const unavailable = capability?.available === false;
  const matches = (p: (typeof PRESETS)[number]) =>
    p.members === draft.members && p.epochs === draft.epochs && p.seeds.length === draft.seeds.length && p.seeds.every((s, i) => draft.seeds[i] === s);
  const recommendedNow =
    draft.members === RECOMMENDED.members && draft.epochs === RECOMMENDED.epochs && draft.engine === RECOMMENDED.engine && matches(PRESETS[1]);
  const command = [
    "python experiment.py",
    draft.engine === "rehearsal" ? `--device cpu --pace ${draft.pace}` : "--device cuda",
    `--agents ${draft.members}`,
    `--epochs ${draft.epochs}`,
    `--seeds ${draft.seeds.join(" ")}`,
  ].join(" ");

  return (
    <section className="hud-card setup" aria-label="New simulation">
      <header className="setup__head">
        <div>
          <p className="eyebrow">NEW SIMULATION</p>
          <h2>
            Define the town, <em className="serif">then open the bar</em>
          </h2>
          <p className="setup__lede">The world below is staged. Nothing runs until you press Begin.</p>
        </div>
        {canClose && (
          <button type="button" className="btn btn--ghost" onClick={onClose} aria-label="Close setup">
            ✕
          </button>
        )}
      </header>

      <div className="presets" role="group" aria-label="Presets">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            className="preset"
            aria-pressed={matches(preset)}
            onClick={() => onDraft({ ...draft, members: preset.members, epochs: preset.epochs, seeds: [...preset.seeds] })}
          >
            <span className="preset__label">
              {preset.label}
              {preset.recommended && <em>★ RECOMMENDED</em>}
            </span>
            <span className="preset__nums">
              {preset.members} members · {preset.epochs} nights · {preset.seeds.length} seed{preset.seeds.length > 1 ? "s" : ""}
            </span>
            <span className="preset__blurb">{preset.blurb}</span>
          </button>
        ))}
      </div>

      <Dial
        id="setup-members"
        label="MEMBERS"
        unit="agents"
        value={draft.members}
        min={LIMITS.members.min}
        max={LIMITS.members.max}
        sliderMax={LIMITS.members.max}
        hint="Recommended 100"
        marks={[
          { value: COMFORT_THRESHOLD, label: `COMFORT LINE ${COMFORT_THRESHOLD}`, tone: "line" },
          { value: RECOMMENDED.members, label: "★ 100", tone: "star" },
        ]}
        onChange={(members) => onDraft({ ...draft, members })}
      />

      <Dial
        id="setup-nights"
        label="NIGHTS"
        unit="epochs"
        value={draft.epochs}
        min={LIMITS.epochs.min}
        max={LIMITS.epochs.max}
        sliderMax={200}
        hint="Recommended 50"
        marks={[
          { value: 20, label: "MIN 20", tone: "line" },
          { value: RECOMMENDED.epochs, label: "★ 50", tone: "star" },
        ]}
        onChange={(epochs) => onDraft({ ...draft, epochs })}
      />

      <div className="setup__field">
        <div className="dial__head">
          <span className="setup__label">ENGINE</span>
          {capability && (
            <span className="dial__hint" data-tone={capability.available === false ? "fail" : capability.available ? "good" : "idle"}>
              {capability.available === null ? "checking…" : capability.detail}
            </span>
          )}
        </div>
        <div className="engines" role="radiogroup" aria-label="Engine">
          {ENGINES.map((engine) => {
            const cap = launcher.status?.capabilities?.[engine.id];
            return (
              <button
                key={engine.id}
                type="button"
                role="radio"
                aria-checked={draft.engine === engine.id}
                aria-disabled={engine.lock ? true : undefined}
                data-lock={engine.lock}
                className="engine"
                onClick={() => {
                  if (!engine.lock) onDraft({ ...draft, engine: engine.id });
                }}
              >
                <span className="engine__dot" data-tone={engine.lock || cap?.available === false ? "fail" : cap?.available ? "good" : "idle"} aria-hidden="true" />
                <span className="engine__title">{engine.title}</span>
                <span className="engine__sub">{engine.sub}</span>
                {engine.lock && (
                  <span className="engine__lock" role="tooltip">
                    {engine.lock}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        {draft.engine === "rehearsal" && (
          <label className="pace">
            <span>PACE</span>
            <input
              type="range"
              min={0.2}
              max={5}
              step={0.1}
              value={draft.pace}
              onChange={(event) => onDraft({ ...draft, pace: Number(event.target.value) })}
              style={{ ["--fill" as string]: `${((draft.pace - 0.2) / 4.8) * 100}%` }}
            />
            <strong>{draft.pace.toFixed(1)}s</strong>
            <em>per night</em>
          </label>
        )}
      </div>

      <div className="setup__field">
        <div className="dial__head">
          <span className="setup__label">SEEDS</span>
          <span className="dial__hint">each seed runs control + broadcast</span>
        </div>
        <SeedEditor seeds={draft.seeds} onChange={(seeds) => onDraft({ ...draft, seeds })} />
      </div>

      <dl className="setup__summary">
        <div>
          <dt>TRIALS</dt>
          <dd>{estimate.trials}</dd>
        </div>
        <div>
          <dt>DECISIONS</dt>
          <dd>{estimate.decisions.toLocaleString()}</dd>
        </div>
        <div>
          <dt>{estimate.calibrated ? "EST. TIME" : "ROUGH TIME"}</dt>
          <dd>≈ {fmtDuration(estimate.seconds)}</dd>
        </div>
      </dl>

      <ul className="advice">
        {advice.map((item) => (
          <li key={item.text} data-tone={item.tone}>
            <span className="advice__icon" aria-hidden="true">
              {item.tone === "good" ? "✓" : "!"}
            </span>
            <span>{item.text}</span>
          </li>
        ))}
      </ul>

      {offline ? (
        <div className="setup__offline" role="status">
          <p>The launcher is offline (open this page through <code>npm run dev</code>). Start the run from a terminal instead:</p>
          <code className="setup__command">{command}</code>
        </div>
      ) : (
        <p className="setup__fineprint">
          Begin moves the current contents of <code>{launcher.status?.dataDir ?? "outputs/experiment"}</code> into an archive folder beside it, then
          starts <code>experiment.py</code>.
        </p>
      )}

      {(launcher.error || errors.length > 0) && (
        <p className="setup__error" role="alert">
          {launcher.error ?? errors[0]}
        </p>
      )}

      <footer className="setup__actions">
        <button type="button" className="btn btn--ghost" onClick={() => onDraft({ ...RECOMMENDED, seeds: [...RECOMMENDED.seeds] })} disabled={recommendedNow}>
          ↺ RECOMMENDED
        </button>
        <button
          type="button"
          className="btn btn--go btn--big"
          data-armed={armed}
          disabled={!config || offline || unavailable || launcher.pending}
          onClick={() => {
            if (!config) return;
            if (!armed) {
              setArmed(true);
              return;
            }
            setArmed(false);
            onBegin(config);
          }}
        >
          {launcher.pending
            ? "OPENING THE BAR…"
            : armed
              ? `CONFIRM · ${estimate.trials} TOWNS · ≈ ${fmtDuration(estimate.seconds)}`
              : "▶ BEGIN SIMULATION"}
        </button>
      </footer>
    </section>
  );
}
