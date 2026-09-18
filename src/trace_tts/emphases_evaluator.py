import ast
import contextlib
import importlib.metadata
import importlib.util
from pathlib import Path
import re
import sys
import time
import types
import librosa
import pypar
import soundfile as sf
import torch
import torchaudio

def module(name, path):
    obj = types.ModuleType(name)
    obj.__path__ = [str(path)]
    obj.__file__ = str(path / '__init__.py')
    sys.modules[name] = obj
    return obj

def full_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    chosen = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in chosen} != set(names):
        raise RuntimeError(f'Upstream definition mismatch: {path}, {names}')
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), 'exec'), namespace)

def build_model(weight):
    if importlib.metadata.version('emphases') != '0.0.2':
        raise RuntimeError('This loader is restricted to emphases==0.0.2')
    source = Path(importlib.util.find_spec('emphases').origin).parent
    e = module('emphases', source)
    defaults = full_module('emphases.config.defaults', source / 'config/defaults.py')
    e.__dict__.update({k: v for k, v in vars(defaults).items() if k.isupper()})
    static = full_module('emphases.config.static', source / 'config/static.py')
    e.__dict__.update({k: v for k, v in vars(static).items() if k.isupper()})
    expected = {'ARCHITECTURE': 'convolution', 'METHOD': 'neural', 'LOSS': 'bce', 'MEL_FEATURE': True, 'PITCH_FEATURE': False, 'LOUDNESS_FEATURE': False, 'PERIODICITY_FEATURE': False, 'NORMALIZE': False, 'DOWNSAMPLE_LOCATION': 'intermediate', 'DOWNSAMPLE_METHOD': 'sum'}
    for k, v in expected.items():
        if getattr(e, k) != v:
            raise RuntimeError(f'Unsupported configuration: {k}')
    e.convert = full_module('emphases.convert', source / 'convert.py')
    e.data = module('emphases.data', source / 'data')
    e.data.preprocess = module('emphases.data.preprocess', source / 'data/preprocess')
    mel = module('emphases.data.preprocess.mels', source / 'data/preprocess')
    mel.__dict__.update(emphases=e, torch=torch, librosa=librosa)
    definitions(source / 'data/preprocess/mels.py', ['from_audio', 'linear_to_mel'], mel.__dict__)
    e.data.preprocess.mels = mel
    e.data.preprocess.__dict__.update(emphases=e, torch=torch)
    definitions(source / 'data/preprocess/core.py', ['from_audio'], e.data.preprocess.__dict__)
    core = {'emphases': e, 'torch': torch, 'torchaudio': torchaudio, 'contextlib': contextlib}
    names = ['postprocess', 'preprocess', 'downsample', 'resample', 'inference_context']
    definitions(source / 'core.py', names, core)
    for name in names:
        setattr(e, name, core[name])
    e.model = module('emphases.model', source / 'model')
    conv = full_module('emphases.model.convolution', source / 'model/layers/convolution.py')
    e.model.__dict__.update(emphases=e, Convolution=conv.Convolution)
    definitions(source / 'model/layers/__init__.py', ['Layers'], e.model.__dict__)
    model_module = full_module('emphases.model.core', source / 'model/core.py')
    model = model_module.Model()
    checkpoint = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return (e, model)

def normalized_words(text):
    return re.findall("[a-z0-9]+(?:'[a-z0-9]+)*", text.lower().replace('’', "'"))

def score_file(e, model, wav, grid, lab):
    alignment = pypar.Alignment(grid)
    samples, rate = sf.read(wav, dtype='float32', always_2d=True)
    if samples.shape[1] != 1:
        raise ValueError('Unexpected stereo input: define handling before formal evaluation')
    audio = torch.from_numpy(samples.T.copy())
    silence = {pypar.SILENCE, '<eps>', '', 'sil', 'sp'}
    lexical = [i for i, w in enumerate(alignment) if str(w).lower() not in silence]
    words = [str(alignment[i]) for i in lexical]
    reference = lab.read_text(encoding='utf-8')
    if normalized_words(' '.join(words)) != normalized_words(reference):
        raise ValueError(f'Word order/cardinality differs from reference: {wav}')
    start_time = time.perf_counter()
    chunks = []
    for features, bounds in e.preprocess(alignment, audio, rate):
        lengths = torch.tensor([features.shape[-1]], dtype=torch.long)
        word_lengths = torch.tensor([bounds.shape[-1]], dtype=torch.long)
        with e.inference_context(model):
            logits = model(features, lengths, bounds, word_lengths)
        chunks.append(e.postprocess(logits.detach()[0]).float())
    if not chunks:
        raise ValueError('No inference output; upstream preprocessing failed')
    scores = torch.cat(chunks, 1)
    if tuple(scores.shape) != (1, len(alignment)):
        raise ValueError('Inference lost words or returned unexpected dimensions')
    if not torch.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Non-finite/out-of-range prominence')
    return {'id': wav.stem, 'waveform': str(wav), 'alignment': str(grid), 'transcript': reference, 'words': words, 'scores': [scores[0, i].item() for i in lexical], 'n_words': len(words), 'n_alignment_entries_including_silence': len(alignment), 'audio_seconds': len(samples) / rate, 'scoring_seconds': time.perf_counter() - start_time}
