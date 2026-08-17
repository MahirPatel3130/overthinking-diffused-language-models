# Experiment 2: answer trajectories and remasking

Owner: Rishab. Dataset: GSM8K. Model: LLaDA-8B-Instruct, revision `6059b30`.

This package answers two questions about what happens to the answer during
diffusion inference:

- **2A.** As the model denoises, does the extracted answer change, and when it
  does, is that the model revising its answer or an artifact of reading
  half-written text?
- **2B.** Committed tokens are frozen under standard decoding. If we unfreeze the
  answer tokens and let the model re-predict them, does accuracy improve?

## Why 2A needed a flip-cause taxonomy

The natural way to measure answer stability is to extract an answer from the
decoded text at every denoising step and count how often it changes. Doing that
naively reports that most problems are unstable: at LLaDA's standard config, 77%
of ever-correct problems appear to become correct, then stop being correct.

Almost all of that is measurement error. Two things produce false flips:

- **`partial_commit`** — the number itself is being written across several steps,
  and the decoder does not write digits left to right. `14` becomes `104`,
  `55` becomes `595`, `72000` becomes `720000`. The extracted answer changes
  because the number is incomplete, not because the model changed its mind.
- **`extractor_jump`** — a number is committed elsewhere in the text (often in the
  reasoning body or a partially-written closing sentence) and the answer regex,
  which takes the last match, latches onto it.

Each step pair is labelled `C` (correct), `W` (wrong), or `N` (nothing
extractable), and every `C->W` and `W->C` transition is assigned a cause by
diffing the two decoded strings.

Results:

| config | total flips | partial_commit | extractor_jump | other |
| --- | --- | --- | --- | --- |
| fast (n=1319) | 612 | 413 | 142 | 57 |
| std (n=200) | 560 | 442 | 94 | 24 |

All 81 `other` flips were inspected by hand. Every one but a single case is also
a digit insertion, differing only in that the digit lands mid-number rather than
at either end, which the current prefix/suffix rule does not catch. The last case
(std problem 373, `240000` -> `4`) is an `extractor_jump` onto a `$240,000`
figure being written character by character.

**Zero genuine regressions across 1519 problems and two configs.** This is
consistent with the mechanism: standard masked-diffusion decoding excludes
committed tokens from future prediction, so the model cannot revise a committed
answer. The point of the taxonomy is that the naive measurement says otherwise.

**Known limitation.** The cause classifier tests whether the old answer is a
prefix or suffix of the new one. Mid-number insertions therefore fall through to
`other`. Replacing this with an in-order subsequence check moves roughly 80 flips
into `partial_commit` and does not change any conclusion.

## 2B: remasking

Each problem is run twice with identical seed and config. Run A is normal
decoding. Run B is identical until the model has committed an explicit
`Final answer: N`, at which point those number tokens are reset to `[MASK]` and
re-predicted with the rest of the generated text as context.

**Result on 1319 GSM8K problems at the fast config:** 48.07% -> 49.51%, a gain of
1.44 points. The intervention fired on 77% of problems and changed the answer 15%
of the time it fired. Discordant pairs: 47 helped, 28 hurt, McNemar exact
two-sided p = 0.037.

### The commit check is load-bearing

The intervention only fires when the entire region from the `Final answer` marker
through one token past the number contains no masks, and the number's own token
positions are contiguous.

Without that check the intervention fires on half-written numbers. An earlier
version remasked a lone `0` at step 11 while the reasoning still read `1. **D`,
and that `0` was one digit of a number on its way to becoming `595`. Erasing a
digit mid-write and calling the result "the model reconsidered" would have
produced a much larger and entirely fake effect. Anyone modifying
`locate_final_answer_positions` should re-read an intervention log before
trusting the numbers.

### Confidence gating

Median minimum commit-time confidence on the remasked tokens:

| outcome | n | median confidence |
| --- | --- | --- |
| stayed correct | 438 | 0.819 |
| remasking fixed | 47 | 0.569 |
| remasking broke | 28 | 0.647 |
| stayed wrong | 507 | 0.639 |

Confidently committed answers were already correct, so intervening on them can
only cause damage. Gating the intervention on commit confidence can be simulated
offline from the saved pairs: when the gate blocks, the outcome is Run A's; when
it allows, the outcome is Run B's.

Gating below 0.75 gives 49.89% (+1.82 pp) while firing on 45% fewer problems than
always intervening, at p = 0.003.

That threshold was selected on the same data it was evaluated on. Tuning on even
problem ids and evaluating on odd ids gives +0.91 pp (49.47% -> 50.38%, p = 0.377).
**Quote the held-out number.**

## Config caveat

Two configs appear in the results:

| | steps | block | tokens/step | GSM8K accuracy |
| --- | --- | --- | --- | --- |
| fast (experiment 1's setting) | 64 | 256 (single) | 4 | 48.1% |
| std (LLaDA's recommended) | 256 | 32 | 1 | 80.5% |

All 2B numbers above are at the fast config. Before any of this goes in a paper,
the remasking result needs to be reproduced at the std config.

The two configs also disagree about when the answer appears: median step 13/64 at
fast config with 204 of 256 tokens still masked, versus median step 226/256 at std
config. The second is not the model deliberating longer. With 32-token blocks the
answer span cannot be unmasked before step 225, so it commits 1.5 steps after it
first becomes possible. **In both configs the answer commits as early as the
schedule allows;** what differs is how much reasoning has been written by then.

## Layout

```
experiments/experiment2/
  run_2b.py                  paired normal vs remasking runs (needs a GPU)
  extractor.py               verbatim copy of experiment 1's extractor, EXTRACTOR_VERSION 1
  analyze_transitions.py     2A transition + flip-cause analysis (offline)
  confidence_gate_sweep.py   offline threshold sweep over saved pairs
  make_fixtures.py           synthetic trajectories covering each flip case
  test_analysis.py           asserts the classifier labels each fixture correctly
```

`extractor.py` is duplicated from experiment 1 rather than imported, so that a
change there cannot silently alter these results. If experiment 1's extractor
changes, this copy must be updated deliberately and everything rerun.

## Reproducing

Experiment 1 trajectories (produces the input for 2A):

```bash
CUDA_VISIBLE_DEVICES=0 EXP1_STAGE=full EXP1_NUM_GPUS=1 \
  papermill experiment1_gsm8k.ipynb out_full.ipynb
```

2A analysis (no GPU):

```bash
PYTHONPATH=. python experiments/experiment2/analyze_transitions.py \
  --trajectory-dir artifacts/experiment1_gsm8k/trajectories
```

2B paired runs (~2 hours on one A100 for 1319 problems):

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/experiment2/run_2b.py --stage smoke --limit 5
CUDA_VISIBLE_DEVICES=0 python experiments/experiment2/run_2b.py --stage full
python experiments/experiment2/run_2b.py --analyze-only
```

Offline gate sweep:

```bash
python experiments/experiment2/confidence_gate_sweep.py
```

Artifacts are gzipped JSON, one file per problem, written atomically. Reruns skip
problems that already have a valid artifact with a matching config hash, so a
crashed run resumes by reissuing the same command. Changing any value in
`SCIENTIFIC_CONFIG` changes the hash and invalidates existing artifacts by design.

Run the smoke stage and read an intervention log before any full run. Both bugs
found so far were visible in the first three intervention logs and invisible in
the summary statistics.

## Open items

1. Confidence-gated remasking at the std config. Decides whether this is a method
   or a probe.
2. A compute-matched baseline — remasking costs extra forward passes, so the
   comparison must be against running more denoising steps, not against fewer.
3. MATH500 (boxed-answer extraction, coordinated with Mukarramah).
4. Subsequence fix for the flip-cause classifier.
