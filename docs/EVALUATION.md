# Evaluation interfaces

Evaluation uses frozen WhiStress, emphases, and UTMOS models, together with
held-out Expresso word-level prominence scores. These are automatic evaluations,
not human listening judgments. No evaluator is fitted to controller outputs.

## WhiStress Pair-Contrast and Pair-Correct

`whistress_evaluator.FrozenWhiStressClient` takes local official source, base
model, add-on weights, and a device. `predict_pairs` accepts the waveform,
sampling rate, and transcription. Map detector words to the same canonical word
indices used by the targets.

For a CAST endpoint, Pair-Contrast succeeds if its intended target is detected as
stressed and its competing target is not. Pair-Correct succeeds only when that
criterion holds in both directions of the pair. P-Con averages the endpoint
decisions; P-Corr averages the paired decisions. Each cell contains 113 pairs and
226 endpoints. Incompatible transcripts or detector failures are missing
observations rather than successful negative detections.

`metrics.switching_outcomes` derives the percentages of pairs succeeding in zero,
one, and two directions from those same endpoint decisions. Consequently,
`P-Con = f2 + f1 / 2` and `P-Corr = f2` use identical input judgments and
aggregation. Native uses the target-free host path; target labels remain part of
evaluation, not native synthesis conditioning.

## emphases margin

`emphases_evaluator.build_model` takes a local checkpoint.
`score_file` accepts the model interface, model, waveform path, TextGrid path,
and transcript path. The wrapper uses the pinned emphases distribution's mel-only
CNN and feature configuration.

For outputs requesting target a and target b, respectively, compute:

```text
delta_P = ((p_a[a] - p_a[b]) + (p_b[b] - p_b[a])) / 2
```

Average across the CAST pairs. This is a continuous prominence-score margin, not
a percentage.

## Expresso RankAcc

Each held-out Expresso endpoint supplies its intended target index and the full
word-score vector. `metrics.rank_correct` returns success only if the target
score is strictly greater than every other word's score. Ties fail. Average over
the 30 endpoint entries in each cell.

`expresso_transform.endpoint_prominence(raw, observed, support, scales, direction)`
is an acoustic word-score utility. It accepts word-by-cue descriptors and masks,
positive TRAIN-derived scales, and a TRAIN-derived unit direction. It uses
within-endpoint word centering and projects the scaled descriptors onto that
direction. It does not subtract a mean across generated alternatives. Missing
supported coordinates prevent a completed acoustic score.

The scoring entry accepts explicit `word_scores`, so the score-extraction
configuration, canonical alignment, and source endpoint association remain
inputs to the evaluation pipeline.

## UTMOS

`utmos_evaluator.UTMOS22` wraps the official local `score.Score` interface.
Provide its checkpoint and use the same scoring sample rate for all systems.
Average predicted MOS over all 226 CAST output entries in each cell.

## Aggregation entry

```bash
python scripts/score.py --input YOUR_OBSERVATIONS.json --output YOUR_SUMMARY.json
```

Each input file represents one backbone, one system, one training seed, and one
speaker prompt under oracle target specification. Its JSON contains:

- `cast`: 113 rows with `pair_id`, `target_a`, `target_b`, `stressed_a`,
  `stressed_b`, `prominence_a`, `prominence_b`, `utmos_a`, and `utmos_b`.
- `expresso`: 30 rows with `endpoint_id`, `target`, and `word_scores`.

Targets are zero-based integer word indices. `stressed_a` and `stressed_b`
are lists of unique detected indices, not word-level binary-label arrays.
An empty list denotes a completed detector call with no stress detections;
`null` denotes unavailable output. Prominence arrays and `word_scores` contain
numeric values in complete canonical word order. UTMOS fields are numeric
predictions or `null`. Numeric values must not be quoted strings.

Row IDs are unique within each input and refer to the frozen endpoint registry.
Do not combine different seed/prompt cells into one scoring input. Missing or
non-finite observations leave the affected metric `null`; complete observations
are not silently reweighted to a smaller denominator.

`metrics.aggregate_seeds` first averages the four prompts within each seed, then
computes the mean and sample standard deviation across three distinct training
seeds. Generation seeds do not replace independently trained seeds.

## Development gain

`calibration.select_gain` uses a declared positive candidate grid on
`split='dev'`. Select the maximum development RankAcc, resolving ties toward
the smaller gain, and keep the selected per-backbone value fixed for evaluation.
The inference interface also accepts zero gain for the native bypass.

The generation path ends at the frozen decoder/vocoder. Metric extraction reads
the generated waveform without pitch, time, or loudness editing. Feature-curve
display operations are separate from synthesis and scoring inputs.
