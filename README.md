# TRACE: Target-Relative Acoustic Contrast Encoding for Prosodic Stress Control in Text-to-Speech

This repository contains the implementation of TRACE, a framework for controllable word-level prosodic stress in text-to-speech (TTS).

## Audio samples

The examples below use the same sentence, speaker prompt, and generation seed within each comparison. Only the requested stress target or control method changes.

### Main comparison — Bidirectional stress switching

CosyVoice3 · WordVoice and TRACE

#### Target: MORAL

She defended the **MORAL** theory in the seminar.

**WordVoice**

https://github.com/user-attachments/assets/63057583-8f76fb0b-e4ca-498f-a11f-4e1db1577408

**TRACE**

https://github.com/user-attachments/assets/63057574-1e3fff8c-1637-4236-9c77-39014d7fad12

#### Target: THEORY

She defended the moral **THEORY** in the seminar.

**WordVoice**

https://github.com/user-attachments/assets/63057590-229c14bd-3112-4d4f-be09-b666dc250065

**TRACE**

https://github.com/user-attachments/assets/63057578-14f41618-1b0f-4692-8fa8-a8c011b72e7a

### Additional stress pairs — Stress control across sentence structures

CosyVoice3 · WordVoice and TRACE

BLUE / VASE, DOCTOR / FAMILY, and FRIDAY / MORNING are illustrative examples for listening. They are separate from the quantitative evaluation results.

#### Pair 01: BLUE / VASE — Local contrast

She placed the **blue vase** beside the window.

##### Target: BLUE

She placed the **BLUE** vase beside the window.

**WordVoice**

https://github.com/user-attachments/assets/63057559-cb383001-3e08-44f0-b9fe-fd0c7f6080ff

**TRACE**

https://github.com/user-attachments/assets/63057545-9f19cc10-c3b9-4340-b273-adea3e2c4da6

##### Target: VASE

She placed the blue **VASE** beside the window.

**WordVoice**

https://github.com/user-attachments/assets/63057564-062db153-8266-4752-99f8-c8320f3e3e59

**TRACE**

https://github.com/user-attachments/assets/63057554-58d4a2f4-933f-4b81-9c33-b013dc3522db

#### Pair 02: DOCTOR / FAMILY — Long-range contrast

The **doctor** discussed the treatment with the **family**.

##### Target: DOCTOR

The **DOCTOR** discussed the treatment with the family.

**WordVoice**

https://github.com/user-attachments/assets/63057601-54c7ae56-200b-4166-a4e1-738ee8993296

**TRACE**

https://github.com/user-attachments/assets/63057594-da76fce7-aeb9-467c-83aa-328a5b06a273

##### Target: FAMILY

The doctor discussed the treatment with the **FAMILY**.

**WordVoice**

https://github.com/user-attachments/assets/63057604-628aaa12-0663-4324-811b-18a8960cacd7

**TRACE**

https://github.com/user-attachments/assets/63057596-0b0d4194-72be-4d8f-a876-edb9940559c2

#### Pair 03: FRIDAY / MORNING — Phrase-final contrast

They scheduled the meeting for **Friday morning**.

##### Target: FRIDAY

They scheduled the meeting for **FRIDAY** morning.

**WordVoice**

https://github.com/user-attachments/assets/63057645-eba28735-1954-477b-b584-389ef828a9d6

**TRACE**

https://github.com/user-attachments/assets/63057634-ef75566b-da1c-4c67-8096-049166ef8510

##### Target: MORNING

They scheduled the meeting for Friday **MORNING**.

**WordVoice**

https://github.com/user-attachments/assets/63057652-69800644-ccbd-4ccd-a837-a5c891020b18

**TRACE**

https://github.com/user-attachments/assets/63057640-66c69221-2923-4659-b415-a053f86a443a

### Across backbones — TRACE on three TTS systems

The same target pair is used throughout. The CosyVoice3 TRACE samples appear in the main comparison above.

#### F5-TTS · TRACE

**MORAL target**

https://github.com/user-attachments/assets/63057623-b1474f76-6703-4da1-8298-8ea35d6db157

**THEORY target**

https://github.com/user-attachments/assets/63057626-5fdd60df-4c34-4c6c-9aa5-87f79c789641

#### dots.tts-soar · TRACE

**MORAL target**

https://github.com/user-attachments/assets/63057611-808d5801-23a3-4aaa-bd35-f91b9c07802c

**THEORY target**

https://github.com/user-attachments/assets/63057618-167a530c-610c-43e2-a117-676fd912a269

Current TTS systems can produce natural and expressive speech, but explicitly specifying a stress target does not always lead to reliable prominence realization. TRACE addresses this gap by modeling stress as a target-relative prosodic configuration rather than a set of absolute acoustic values for the target word. It learns from matched renditions with the same text and speaker but different stress targets, capturing how pitch, energy, and duration are reorganized across the utterance.

TRACE combines an ordered relational prosody planner, target-anchored global relaxation, and a command-gated residual adapter on top of a frozen TTS backbone. At inference time, TRACE uses the native TTS state and the selected stress target, without requiring paired reference speech. Under oracle target specification, TRACE consistently improves bidirectional stress switching across CosyVoice3, F5-TTS, and dots.tts-soar on the CAST benchmark while maintaining competitive predicted naturalness.

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
by shared TRACE inference and the CosyVoice3 contract wrapper. Its revision-
specific runtime hooks must be implemented against the selected upstream model
internals. [Evaluation](docs/EVALUATION.md) defines the input
observations and metric aggregation. [Adaptations](docs/ADAPTATIONS.md) describes
the control-module scope and ablation interfaces.

`configs/model_example.json` is a configuration schema. Supply concrete model
dimensions and optimization settings in a separate runtime configuration.
`configs/protocol.json` records the system matrix, data partitions, and evaluation
conventions. 

## Hyperparameters

The TRACE experiments reported in the paper use the following hyperparameters for target-relative projection, global prosody relaxation, command sensitivity, and signed counterfactual refinement.

| Parameter | Runtime key | Value | Description |
|---|---|---:|---|
| $\lambda_u$ | `lambda_missing` | 0.25 | Regularization weight for unobserved coordinates in the target-relative projection objective. |
| $\lambda_r$ | `lambda_r` | 2.0 | Smoothing strength in the target-anchored global prosody relaxation. |
| $\tau_b$ | `tau_b` | 0.7 | Boundary attenuation coefficient controlling how prosodic boundaries reduce smoothing across adjacent words. |
| $\lambda_s$ | `lambda_swap` | 0.5 | Weight of the command-sensitivity margin loss. |
| $\gamma$ | `gamma` | 2.0 | Time-weighting exponent in the command-sensitivity loss. |
| $\mu$ | `swap_margin` | 0.25 | Margin used in the command-sensitivity objective. |
| $\lambda_p$ | `lambda_p` | 1.0 | Weight of the signed projection penalty in counterfactual refinement. |
| $\lambda_{cf}$ | `lambda_cf` | 0.5 | Weight of the counterfactual refinement term. |
| $m$ | `margin` | 1.0 | Projection margin used in signed refinement. |

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
Model weights and datasets have independent terms. `CITATION.cff` provides the software citation; upstream references are in `docs/REFERENCES.bib`.
