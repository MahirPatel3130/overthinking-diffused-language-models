# Experiment 2 — LLaDA-8B-Instruct on MATH-500

Does an "overthinking" analogue exist in masked diffusion language models? Specifically:
during 64 denoising steps, does LLaDA reach a correct answer and then revise it into a
wrong one (C→W)?

Model: `GSAI-ML/LLaDA-8B-Instruct` · Benchmark: MATH-500 (n=500) · Hardware: A100-SXM4-40GB, bf16
Config: `steps=64, gen_length=256, block_length=32` (8 blocks × 8 steps)

## Layout

```
src/
  grading.py        boxed-answer extraction + math_verify LaTeX equivalence grading
  trajectory.py     patched LLaDA sampling loop, records decoded state at every global step
  transitions.py    C/W/N label sequences, strict and post_hoc transition counting
  intervention.py   2B remask-and-repredict: detects committed \boxed{} span, remasks, repredicts
notebooks/
  experiment2a_a100.ipynb    trajectory observation run (with saved outputs)
  experiment2b_a100.ipynb    paired intervention run (with saved outputs)
results/
  exp2a_transition_summary.txt   strict + post_hoc transition counts
  exp2a_wc_flips.txt             the 15 W→C flips with step index and label sequence
  exp2b_paired_summary.txt       paired accuracy, McNemar, confidence by transition class
```

## Experiment 2A — trajectory observation

| | strict | post_hoc |
|---|---|---|
| final accuracy | 28.8% | 28.8% |
| ever correct | 28.8% | 28.8% |
| W→C | 145 | 15 |
| C→W | 1 | 0 |
| C→C | 938 | 939 |
| W→W | 30,416 | 1,647 |
| ended wrong | 0.0% | 0.0% |

**No overthinking analogue.** Zero C→W under post_hoc, one under strict, out of 31,500
transitions. `ever_correct_pct == final_accuracy_pct` — no problem passes through a correct
state and leaves it. This also means **zero oracle early-stopping headroom**: accuracy cannot
be gained by truncating denoising.

**Trajectories are near-frozen.** 146 of 31,500 strict transitions involve any state change (0.46%).

**Correctness resolves very late.** 939 C→C transitions across 144 correct problems ≈ 7.5 steps
in a correct state, placing first-correct emergence around step 56–57 of 64.

**Use post_hoc, not strict.** Under strict, unextractable early states are scored wrong,
inflating W→W to 30,416 and making "stuck wrong for 57 steps" look like a finding when it is
mostly a property of the scoring function. post_hoc restricts counting to steps where the
answer span is extractable.

### On the 15 W→C flips

Partially resolved by step index (see `results/exp2a_wc_flips.txt`):

- **8 of 15 flip at step 63**, the final step, and are consistent with partial-commit
  artifacts — the answer only becomes extractable at the last step, so the preceding W is a
  parse failure rather than a genuine wrong answer.
- **7 of 15 flip earlier** (steps 15, 30, 39, 47, 58, 59, 61) and hold C through to the end.
  These look like genuine wrong→right revisions, not artifacts.

Not yet verified against the raw per-step decoded text.

## Experiment 2B — remask-and-repredict intervention

```
n = 500
normal   0.2880
remasked 0.2940   (+0.60 pp)
intervened on 397
helped 5  hurt 2  McNemar p = 0.4531
{'C->C': 142, 'W->W': 253, 'unparsed': 98, 'W->C': 5, 'C->W': 2}
```

Median minimum commit confidence by class:

| class | n | median min-conf |
|---|---|---|
| C→C | 140 | 0.937 |
| C→W | 2 | 0.749 |
| W→W | 249 | 0.725 |
| W→C | 5 | 0.443 |

**Underpowered null, not a negative result.** +0.6pp is 3 problems; 7 discordant pairs gives
almost no power. Direction is consistent with the GSM8K result (+1.44pp, p=0.037) but
significance is not supported at n=500. Detecting an effect this size at this discordance
rate needs roughly 1,700 intervened problems.

**The confidence structure is the more interesting replication.** The ordering
C→C ≫ C→W > W→W > W→C mirrors GSM8K, with W→C lowest at 0.443 — low-confidence commits are
where intervention has value. Fragile at n=5 and n=2, but it now holds on a second benchmark.

## Known issues

1. **unparsed = 98 (19.6%).** Nearly one in five pairs has no extractable answer in one or
   both arms. True accuracy sits somewhere in [28.8%, 48.4%] depending on how these resolve.
   If the extractor is missing answers written outside `\boxed{}`, these hide potential
   discordant pairs — and with only 7 total, recovering a few would move the numbers. This is
   likely a bigger lever on the results than the intervention itself.
2. **intervened on 397, not 402.** The gap between 500 − 98 and 397 needs an explanation
   (probably boxed answers that never fully commit).
3. **Two-state scoring.** A three-state scheme (unparsed / wrong / correct) would separate
   "when does the answer become extractable" from "does it change once extractable". Right
   now those questions are fused.
4. **Design comparability with GSM8K.** This intervention fires post-run. If the GSM8K 2B
   loop fires mid-run, that is a caveat on any side-by-side table.

## Reproducing

Environment pins that matter: `transformers==4.49.0`, `tokenizers<0.22` (avoids a
`tie_weights()` signature mismatch with LLaDA's remote code). Use `torch_dtype=`, not
`dtype=`, on that version. Set `"cpu": "0GiB"` in `max_memory` to prevent silent CPU
offloading. bf16 on A100; fp16 if running on T4.

Raw per-problem trajectory JSONs from the 2A run are not included — they were left on the
compute server. Everything above is reproducible from the notebooks.
