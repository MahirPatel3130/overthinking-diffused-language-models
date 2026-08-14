# Experiment 1 — MATH-500 answer-emergence tracking (LLaDA-8B-Instruct)

Answers RQ1: *How early during diffusion inference does the correct answer
emerge, and how stable is it after emergence?*

## Files
- `llada_step_trace.py` — modified LLaDA sampler that snapshots decoded text after every denoising step (batch size 1).
- `math_utils.py` — `\boxed{}` extraction + MATH answer equivalence checking.
- `record_utils.py` — turns a per-step text trace into the per-problem JSON record (first_correct_step, post-arrival stats, etc). Unit-testable without torch.
- `run_experiment1.py` — loads the model + MATH-500, runs the traced sampler per problem, writes `results_math500.jsonl`.
- `analyze_experiment1.py` — aggregates the JSONL into the summary stats, histogram, and results table.

## 1. Environment

You'll need a machine with a CUDA GPU (8B params in bf16 is ~16GB VRAM just
for weights — a single A100/H100/3090-class GPU with 24GB+ is comfortable;
Cornell's G2 cluster or a Colab Pro+ A100 instance both work).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`transformers==4.38.2` is pinned because LLaDA's `trust_remote_code=True`
modeling code was written against that version — newer transformers has
occasionally broken the custom model class.

## 2. Sanity check (no GPU needed)

```bash
python -c "
from record_utils import build_record
r = build_record('p1', '42', 6,
    ['\\\\boxed{35}','\\\\boxed{35}','\\\\boxed{42}','\\\\boxed{42}','\\\\boxed{41}','\\\\boxed{42}'])
print(r)
"
```
This exercises the extraction + bookkeeping logic against the doc's own
worked example (first_correct_step=3, changed_after_arrival=True) with no
model download required.

## 3. Run a small pilot

Start with 5-10 problems before committing GPU-hours to the full 500:

```bash
python run_experiment1.py --n_problems 5 --steps 64 --gen_length 256 --block_length 64 --out pilot.jsonl
```

Notes on the flags:
- `--steps` must be a multiple of `gen_length / block_length`.
- `--gen_length` needs to be long enough to fit a full MATH solution + `\boxed{}` — 256 is a reasonable start; bump to 384-512 for the harder Level 4/5 problems if you see truncated traces.
- `--block_length` controls the semi-autoregressive block size used by LLaDA's sampler; `block_length == gen_length` gives you a single global block (closer to "pure diffusion"), smaller blocks are faster but generate left-to-right in chunks.

Watch the per-problem print lines — if `first_correct_step` is `None` for
almost everything, `--gen_length` is probably too short (answers are getting
truncated before the model reaches `\boxed{}`).

## 4. Run the full set

```bash
python run_experiment1.py --n_problems 500 --steps 64 --gen_length 256 --block_length 64 --out results_math500.jsonl
```

This can take a while (8B model, 64 forward passes per problem) — consider
running it in a `tmux`/`screen` session or as a batch job on the cluster
rather than in an interactive terminal, since it may run for hours over 500
problems. `run_experiment1.py` flushes to disk after every problem, so it's
safe to `Ctrl+C` and resume by rerunning with `--start_idx` set to how many
you've already completed.

## 5. Analyze

```bash
python analyze_experiment1.py --in results_math500.jsonl --out_dir figs/
```

Prints the mean/median first_correct_step, the "stayed correct / temporarily
wrong / ended wrong" breakdown, and saves:
- `figs/first_correct_step_histogram.png`
- `figs/results_table.csv` (the per-problem table format from the design doc)

## Using VS Code

Yes — VS Code works fine for this. A few things worth setting up:
- **Python extension** (ms-python.python) for linting/debugging.
- **Remote - SSH extension** if you're running on a lab GPU machine or Cornell's cluster rather than your laptop — lets you edit locally and run on the remote box with full IntelliSense, instead of editing over a terminal SSH session.
- If you're on a shared cluster with a job scheduler (Slurm), you'll still submit the actual run as a batch job (`sbatch`) — VS Code Remote-SSH is for editing/debugging, not a substitute for the scheduler.
- The launch config below is handy for the pilot run (`.vscode/launch.json`):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Pilot run (5 problems)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/run_experiment1.py",
      "args": ["--n_problems", "5", "--steps", "64", "--gen_length", "256", "--block_length", "64", "--out", "pilot.jsonl"],
      "console": "integratedTerminal"
    }
  ]
}
```

## Where this fits in the repo

If your group's shared repo already has a structure (e.g. `experiments/`,
`shared/`), drop these files under something like
`experiments/exp1_math500/` and adjust the `--out`/`--out_dir` paths
accordingly — nothing here assumes a particular repo layout. If you tell me
your repo's URL or structure I can tailor the paths and imports to match.
