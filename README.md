# El Farol Bar, with liars

A reference implementation of W. Brian Arthur's El Farol Bar problem (1994), extended with talk.
A town of patrons decides every night whether to go to the bar. Going is good while fewer than 60 show up and bad once it overcrowds.
Each patron is an isolated PyTorch brain: a speaker network that announces "going" or "staying" and an actor DQN that decides what to actually do, so a patron can say one thing and do another.

Every seed runs two towns side by side:

- **control**: nobody can hear anybody.
- **broadcast**: patrons read and write a public feed, so honesty, trust and deception start to matter.

**Live reference:** https://el-farol-rho.vercel.app

## Run it in the browser (any device)

The live site opens on a replay of a PyTorch sweep. Press **New simulation** to train new towns on the visitor's own device, whether that's a phone, tablet or laptop. There's no install and no server.

- **The brains:** `dashboard/src/engine/` is a TypeScript port of the PyTorch brains. It keeps the same layers, initialisation, softmax/ε policy, replay memory, smooth-L1 and MSE losses, target networks, gradient clipping and Adam. It runs in a Web Worker and writes the same files `experiment.py` writes.
- **The statistics:** they come from the real `analyzer.py`, run in the browser by [Pyodide](https://pyodide.org) with numpy and pandas.
- **Reproducibility:** the random number streams differ from PyTorch, so a given seed won't reproduce bit-for-bit. Across seeds the two engines produce the same behaviour statistically.
- **Privacy:** nothing is uploaded, and results last until the tab is closed.

Add `?engine=browser` to a local `npm run dev` URL to use the browser engine there as well.

## Run it on any machine with PyTorch

The brains are small (about 4k parameters each), so they train on a plain CPU in seconds. A CUDA GPU speeds up big towns, but you don't need one.

### 1. Local (Windows, macOS, Linux)

Requires Python 3.11+ and Node 20+.

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere), then:

```bash
pip install -r requirements.txt
npm run setup
npm run dev
```

Open http://localhost:5173 and press **Begin**. The dashboard looks for a Python that has PyTorch, in this order:

1. `ELFAROL_PYTHON`
2. the repo `.venv`
3. `python3` / `python` / `py` on your PATH

On Windows, if none of those has PyTorch, it falls back to a WSL distro as a last resort.

### 2. Docker

The image uses CPU PyTorch by default:

```bash
docker build -t elfarol .
docker run -p 4173:4173 -v elfarol-data:/app/outputs elfarol
```

For an NVIDIA GPU, build with the CUDA wheels instead and run with `--gpus all`:

```bash
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t elfarol:cuda .
```

The container starts on the bundled replay, and **Begin** trains new towns inside it. The image also deploys as-is to any container host (Render, Fly.io, Railway, a VPS). Anyone who can reach that URL can start runs, so put it behind auth if it's public.

### 3. Terminal only

```bash
python experiment.py --agents 100 --epochs 50 --seeds 42 100 2026
python analyzer.py --experiment-dir outputs/experiment
```

`--device auto` is the default. It picks CUDA when present, otherwise CPU, which is all Apple Silicon or a laptop needs.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ELFAROL_PYTHON` | auto | Interpreter that has PyTorch |
| `ELFAROL_DEVICE` | `auto` | Device passed to `experiment.py` (`cpu` or `cuda`) |
| `ELFAROL_TORCH_HOST` | auto | `native` to never use WSL, `wsl` to always use it |
| `ELFAROL_WSL_DISTRO` / `ELFAROL_WSL_PYTHON` | `Ubuntu-24.04` / `~/.venv/bin/python` | Only used by the WSL fallback |
| `ELFAROL_DATA_DIR` | `outputs/experiment` | Where sweeps are written and read |
| `ELFAROL_ALLOWED_HOSTS` | none | Extra hostnames the server accepts (`*` for any) |
| `PORT` | `4173` | Port for `npm start` / Docker |

## Publishing the web reference

The hosted site is a static build with no backend. It opens on whatever sweep is in `dashboard/public/data`, and trains new ones in the browser. To publish a new opening sweep:

```bash
npm --prefix dashboard run snapshot
cd dashboard && npx vercel deploy --prod
```

The snapshot copies `outputs/experiment` into `public/data` and strips local file paths.

## Tests

```bash
pytest
```
