import importlib
import math
import os
from pathlib import Path
import sys

class UTMOS22:

    def __init__(self, source_dir, checkpoint, *, input_sample_rate, device='cpu'):
        self.source = Path(source_dir).resolve(strict=True)
        checkpoint = Path(checkpoint).resolve(strict=True)
        if not (self.source / 'score.py').is_file():
            raise FileNotFoundError('Provide the official UTMOS22 source containing score.py')
        sys.path.insert(0, str(self.source))
        module = importlib.import_module('score')
        if Path(module.__file__).resolve() != self.source / 'score.py':
            raise RuntimeError('A different score module is already imported; use a separate evaluator process')
        previous = Path.cwd()
        try:
            os.chdir(self.source)
            self.scorer = module.Score(ckpt_path=str(checkpoint), input_sample_rate=input_sample_rate, device=device)
        finally:
            os.chdir(previous)
        self.rate, self.device = (input_sample_rate, device)

    def score_file(self, path):
        import torch
        import torchaudio
        waveform, rate = torchaudio.load(str(path))
        if waveform.shape[0] != 1 or waveform.shape[1] == 0 or (not bool(torch.isfinite(waveform).all())):
            raise ValueError('Require nonempty finite mono audio')
        if rate != self.rate:
            waveform = torchaudio.functional.resample(waveform, rate, self.rate)
        with torch.inference_mode():
            raw = self.scorer.score(waveform.to(self.device))
        import numpy as np
        values = np.asarray(raw.detach().cpu() if isinstance(raw, torch.Tensor) else raw).reshape(-1)
        if len(values) != 1 or not math.isfinite(float(values[0])):
            raise ValueError('UTMOS did not return one finite prediction')
        return float(values[0])
