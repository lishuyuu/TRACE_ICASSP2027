# TRACE: Target-Relative Acoustic Contrast Encoding

This repository contains the implementation of TRACE, a framework for controllable word-level prosodic stress in text-to-speech (TTS).

Current TTS systems can produce natural and expressive speech, but explicitly specifying a stress target does not always lead to reliable prominence realization. TRACE addresses this gap by modeling stress as a target-relative prosodic configuration rather than a set of absolute acoustic values for the target word. It learns from matched renditions with the same text and speaker but different stress targets, capturing how pitch, energy, and duration are reorganized across the utterance.

TRACE combines an ordered relational prosody planner, target-anchored global relaxation, and a command-gated residual adapter on top of a frozen TTS backbone. At inference time, only the selected stress target is required, without paired reference speech. Across CosyVoice3, F5-TTS, and dots.tts-soar, TRACE consistently improves bidirectional stress switching on the CAST benchmark while maintaining competitive predicted naturalness.

## Installation

Use Python 3.10-3.12 and install a matching PyTorch/torchaudio build for the host
platform. From the source directory:

```bash
python -m pip install -e .
python -m pip install -e '.[audio]'
python -m pip install -e '.[prominence]'
```

The audio and prominence extras are needed only by their respective interfaces.
Use the selected upstream TTS environment for host model loading and generation.
Installation does not download model weights. See [assets](docs/ASSETS.md) for
upstream sources.

Keep the source tree for the commands below: a wheel installs the `trace_tts`
modules and license notices, while `scripts/`, `configs/`, `data/`, and `docs/`
belong to the source distribution.

## Method modules

| Module | Responsibility |
|---|---|
| `trace_core.py` | Relative coordinates, masked target projection, prominence ordering, acoustic commands, and centered residual targets |
| `trace_networks.py` | Transformer relational planner and temporal residual backbone |
| `global_relaxation.py` | Boundary-weighted global smoothing with zero centering, target anchoring, and prominence constraints |
| `signed_refinement.py` | Same-state ordered-pair signed residual objective |
| `inference.py` | Relaxed plan, shared control gain, flow correction, and exact native bypass |
| `backend_contract.py` | Native state, timing transport, frozen velocity, integration, and decoding interface |
| `trace_audio_features.py`, `trace_phone_clock.py` | Acoustic descriptors, support masks, TRAIN-only scaling, and phoneme-aligned transport |
| `control_embeddings.py`, `cae_reference.py` | Adapted conditioning and activation-editing primitives |
| `metrics.py` | Five metric definitions and prompt/seed aggregation |

## Use

```bash
python scripts/check_splits.py
python scripts/train_planner.py --help
python scripts/score.py --help
```

[Training](docs/TRAINING.md) defines the complete-group input schema and the
planner, adapter-fitting, and refinement entry points.
[Core interfaces](docs/CORE_INTERFACES.md) describes the host operations needed
by shared TRACE inference. [Evaluation](docs/EVALUATION.md) defines the input
observations and metric aggregation. [Adaptations](docs/ADAPTATIONS.md) describes
the control-module scope and ablation interfaces.

`configs/model_example.json` is a configuration schema. Supply concrete model
dimensions and optimization settings in a separate runtime configuration.
`configs/protocol.json` records the system matrix, data partitions, and evaluation
conventions. These files contain no measured scores.

## Hyperparameters

The TRACE experiments reported in the paper use the following hyperparameters for target-relative projection, global prosody relaxation, command sensitivity, and signed counterfactual refinement.

| Parameter | Value | Description |
|---|---:|---|
| $\lambda_u$ | 0.25 | Regularization weight for unobserved coordinates in the target-relative projection objective. |
| $\lambda_r$ | 2.0 | Smoothing strength in the target-anchored global prosody relaxation. |
| $\tau_b$ | 0.7 | Boundary attenuation coefficient that controls how prosodic boundary strength reduces smoothing across adjacent words. |
| $\lambda_s$ | 0.5 | Weight of the command-sensitivity margin loss used to encourage target-dependent command responses. |
| $\gamma$ | 2.0 | Time-weighting exponent in the command-sensitivity loss. |
| $\mu$ | 0.25 | Margin used in the command-sensitivity objective. |
| $\lambda_p$ | 1.0 | Weight of the magnitude/projection penalty in signed refinement. |
| $\lambda_{cf}$ | 0.5 | Weight of the counterfactual signed-refinement objective. |
| $m$ | 1.0 | Projection margin used in signed refinement. |

These settings correspond to the reported experiments unless otherwise specified by an ablation or runtime configuration.

## Software tests

```bash
python -m unittest discover -s tests -v
```

Tests use artificial tensors and interval fixtures. They do not download models,
train TTS systems, or generate speech.

## License and citation

Project-authored code is under MIT. Third-party-derived portions retain their
notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
Model weights and datasets have independent terms. `CITATION.cff` provides the
anonymous software citation; upstream references are in `docs/REFERENCES.bib`.
