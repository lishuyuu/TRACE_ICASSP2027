# Training and inference

## Data preparation

Resolve the Expresso source IDs in `data/splits/expresso_{train,dev,test}.json`
against the upstream recordings. A matched group contains shared text and speaker
identity with different target words. Keep complete groups in their specified
split. CAST is reserved for evaluation. Obtain prompt recordings separately using
`data/prompts.json`.

The acoustic preparation interface accepts finite, one-dimensional floating-point
mono audio at 24,000 Hz. Its MFA/Praat input is a long-format TextGrid containing
word and phone intervals with English ARPAbet phone labels. Convert the input
format explicitly before extraction.

`trace_audio_features` extracts three log-F0 Legendre coefficients, log-RMS
energy, and log-duration. Reliable cue observations define the support mask.
Missing supported values use same-word, same-cue matched-group means without
changing that mask; unsupported values are zero. Word centering and TRAIN-only
scaling form the standardized descriptors. Complete-group centering forms
target-relative contrasts. `trace_core.project_target_maximum` obtains the
zero-centered, target-maximal supervision from these contrasts using the supplied
unobserved-coordinate regularization.

Estimate the prominence direction and scales only from training groups and reuse
them for held-out data. All canonical word indices, phone boundaries, and target
indices must refer to the same token sequence.

## Planner

The Transformer planner combines native word states, speaker conditioning,
target-role embeddings, and position embeddings. Its ordered output has a scalar
prominence branch and an orthogonal cue branch. Fit this output to the projected
supervision using masked Huber loss.

`scripts/train_planner.py` accepts prepared complete-group tensors:

```text
omega: float [5]
groups: list
  split: "train"
  native_word_states: float [N,W,H]
  targets: int64 [N]
  speaker_embeddings: float [N,S]
  valid_words: bool [N,W]
  oracle_coordinates: float [N,W,5]
  observed: bool [N,W,5]
```

N indexes the retained alternative targets in one matched group. Targets must be
distinct valid word positions and padding must be explicit. Prepare projected
`oracle_coordinates` before passing them to the entry point.

```bash
python scripts/train_planner.py --help
```

The entry accepts `--groups`, `--config`, `--seed`, `--steps`, `--lr`,
`--weight-decay`, `--huber-delta`, `--output`, and `--device`.
It samples complete groups uniformly and uses AdamW. Architecture dimensions and
optimization settings are runtime inputs; `configs/model_example.json` describes
their fields.

`--validate-only` checks all supplied groups and settings without optimizer
updates. No output path is required in that mode. A training invocation writes a
new checkpoint containing the planner state, architecture configuration, training
direction, and invocation parameters. Load the state with `strict=True` into the
matching architecture. The checkpoint is for inference; it does not contain
optimizer or random-generator state for exact training resumption.

## Global relaxation

Before constructing acoustic commands, pass each plan through the target-anchored
global relaxation. Boundary strengths come from native inter-word pauses and
punctuation; `tau_b` sets their conversion to propagation weights. `lambda_r`
sets the smoothing penalty. The target row is anchored to its input value while
cue-wise centering and target prominence ordering are retained.

`inference.relax_training_plans` selects oracle plans during warm-up and detached
planner predictions thereafter, then applies the same relaxation. Supply
`oracle`, `prediction`, `targets`, `omega`, `boundary_strengths`, `warmup`,
`lambda_r`, and `tau_b`.

## Residual fitting and signed refinement

A complete-group host batch uses one clock sampled from the group's native and
target-specific clocks. Transport all endpoints and native conditioning
phone-wise to that clock at fixed total frame count. Use shared noise and time,
evaluate the frozen field at each endpoint's interpolation state, and subtract the
complete-group residual mean to form the residual targets.

The residual backbone combines flow state, native conditioning, speaker
information, and time using temporal residual convolutions. Bias-free command
gating and output projection make the correction zero for a zero command.

`training_steps.adapter_fit_step` fits the valid-frame MSE and a sampled
alternative-command margin. Both commands use the same state and target. Supply
the margin weight, time exponent, margin, and flow interpolation settings as
runtime parameters.

`training_steps.adapter_refine_step` retains residual MSE and uses all valid
ordered command pairs at their corresponding anchors. The normalized signed
objective combines smoothed cosine alignment and a unit-threshold Huber
projection penalty. Supply the signed weight, projection weight, projection
margin, and numerical smoothing parameters. Keep the planner and host frozen.

## Generation and calibration

Use `inference.generate_trace` with an instantiated host binding as described in
[core interfaces](CORE_INTERFACES.md). The single scalar `alpha` multiplies both
the relaxed plan and the masked residual correction. The host owns the sampler
and frozen decoder. A zero gain returns the native path.

Declare the gain candidates before evaluation. Select a per-backbone gain using
development RankAcc, with smaller gain resolving equal scores, then keep it
fixed on test inputs. Candidate values, optimizer settings, update counts, and
host sampler settings are supplied by the runtime configuration rather than
chosen by the source package.
