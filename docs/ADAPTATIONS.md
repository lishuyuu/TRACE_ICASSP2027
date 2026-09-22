# Control adaptations and ablations

The comparison matrix contains Native, EEC-adapted, CAE-adapted,
CtrlSpeech-adapted, WordVoice-adapted, and TRACE on each selected host.
Adapted controls share the host's native acoustic representation and frozen
decoder. Their names identify the source mechanisms, not complete original
author systems.

## Control interfaces

| Family | Core operation | Host-side connection |
|---|---|---|
| EEC | Explicit word-level emphasis conditioning | Target-label representation, trainable conditioning projection, and host acoustic training interface |
| CAE | PCA/probe-based target-local activation edit in `cae_reference.py` | Native feature extraction and insertion of the edited state |
| CtrlSpeech | Pitch, loudness, and duration phone embeddings in `control_embeddings.py` | Canonical frame/phone mapping and acoustic conditioning projection |
| WordVoice | Word-attribute embedding and frame modulation in `control_embeddings.py` | Attribute assignment, explicit duration mapping, and native hidden-state modulation |
| TRACE | Relational planner, global relaxation, and gated residual control | Operations in `backend_contract.py` |

The EEC row describes the explicit-conditioning connection required by its host
controller. The supplied conditioning and editing primitives are composed with
host-specific training and generation code. Keep upstream attribution when using
the derived CtrlSpeech and WordVoice modules.

All controllers use the same target information, training-data access, fixed
speaker prompts, and evaluation interfaces. Fit data-dependent transforms on
TRAIN and choose control gains on development data. Do not fit controller
parameters or choose gains using the evaluation models or final CAST pairs.
Native is the unmodified target-free reference path.

## Ablation names

`configs/ablations.json` contains the four component variants:

- w/o target-relative contrast
- w/o group centering
- w/o prominence ordering
- target-local only

The manifest separates the public variant identifier from its implementation
binding. Training and generation use the same selected binding and retain the
same data access, host, and evaluation interface.

For the no-ordering path, remove the target-order constraint from supervision,
planner parameterization, and global relaxation. For target-local control, restrict the acoustic
control to the requested word. The target-relative-contrast and  group-centering
variant identifiers are connected through explicitly supplied variant bindings.
