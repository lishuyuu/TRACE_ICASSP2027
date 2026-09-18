# Core and host interfaces

TRACE's word-level representation, planner, global relaxation, and residual
objective are shared across host models. The acoustic state dimensions, native
conditioning, alignment transport, integration method, and decoder belong to the
host implementation.

## Native and transported state

`backend_contract.NativeState` carries `waveform`, `word_states`, `speaker`,
`word_durations`, `boundary_strengths`, and an opaque host `context`.

`word_states` has shape [words, native channels], `speaker` is a speaker vector,
and `word_durations` contains the positive native word durations.
`boundary_strengths` has one nonnegative entry per adjacent word pair.

`backend_contract.TransportedCondition` carries `flow_condition`,
`adapter_condition`, `word_index`, `nucleus_phase`, `voicing_gate`, `valid_frames`,
and an opaque host `context`.

Boundary strengths are nonnegative values for each adjacent word pair, obtained
from punctuation and native inter-word pauses. Their conversion into propagation
weights is `exp(-tau_b * boundary_strengths)`.

## Host operations

| Operation | Responsibility |
|---|---|
| `freeze` | Freeze the host backbone and native decoder/vocoder |
| `native` | Obtain target-free waveform, native states, alignment, speaker conditioning, and sampling context |
| `transport` | Align native conditioning and vowel geometry to normalized word-duration proportions at the native total frame count |
| `velocity` | Evaluate the frozen acoustic field on the supplied state, time, and transported condition |
| `integrate` | Run the host sampler with the supplied corrected field and native sampling context |
| `decode` | Convert the integrated acoustic state through the frozen native decoder/vocoder |

Implement these operations against the chosen upstream host and pass that object
to `inference.generate_trace`. Model-name strings in the protocol identify host
assets; they do not instantiate bindings or download weights.

## Shared generation

`generate_trace` accepts the host object and keyword arguments `text`, `prompt`,
`seed`, `target`, `planner`, `adapter`, `alpha`, `scales`, `lambda_r`, and `tau_b`.

The planner predicts a five-cue word plan. Global relaxation minimizes its
distance to that plan plus a boundary-weighted adjacent-word smoothing penalty.
The target row stays fixed, cue-wise sums stay zero, and the target remains
maximal along the prominence direction. The input plan and relaxed plan use
standardized coordinates.

The same scalar `alpha` scales the relaxed plan and the residual correction.
Destandardized duration offsets determine the new clock; pitch and energy offsets
form the frame command. Integration uses the frozen host field plus the masked
adapter correction. At `alpha = 0`, generation returns the native waveform with
native timing and conditioning.

The shared path does not edit waveform pitch, loudness, or duration after decoding.
Acoustic feature extraction and display of feature curves are separate operations
from audio generation and metric calculation.
