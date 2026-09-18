from __future__ import annotations
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any, Sequence
import numpy as np
from .evaluator_errors import ContractError, TokenMapError

@dataclass(frozen=True)
class WordLogitSensitivity:
    word: str
    binary_label: int
    class1_softmax: float

def _merge_decoded_values(token_ids: Sequence[int], values: Sequence[float | int], tokenizer: Any) -> list[tuple[str, float]]:
    if len(token_ids) != len(values):
        raise ContractError('Token/value length mismatch')
    special_ids = set((int(value) for value in tokenizer.all_special_ids))
    filtered: list[tuple[str, float]] = []
    for token_id, value in zip(token_ids, values):
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        filtered.append((decoded, float(value)))
    merged: list[tuple[str, float]] = []
    current_word = ''
    current_value = float('-inf')
    for token, value in filtered:
        if token.startswith(' ') or current_word == '':
            if current_word:
                merged.append((current_word.strip(), current_value))
            current_word = token
            current_value = value
        else:
            current_word += token
            current_value = max(current_value, value)
    if current_word:
        merged.append((current_word.strip(), current_value))
    if any((not word for word, _ in merged)):
        raise TokenMapError('WhiStress subtoken merge produced an empty lexical word')
    return merged

def merge_shifted_token_outputs(token_ids: Sequence[int], binary_labels: Sequence[int], class1_softmax: Sequence[float], tokenizer: Any) -> list[WordLogitSensitivity]:
    binary_words = _merge_decoded_values(token_ids, binary_labels, tokenizer)
    score_words = _merge_decoded_values(token_ids, class1_softmax, tokenizer)
    if [word for word, _ in binary_words] != [word for word, _ in score_words]:
        raise ContractError('Binary and score subtoken merges produced different words')
    return [WordLogitSensitivity(word=binary_word, binary_label=int(binary_value), class1_softmax=float(score_value)) for (binary_word, binary_value), (_, score_value) in zip(binary_words, score_words)]

class FrozenWhiStressClient:

    def __init__(self, source_dir: Path, base_model_dir: Path, add_on_weight_dir: Path, device: str) -> None:
        self.source_dir = source_dir.resolve()
        self.base_model_dir = base_model_dir.resolve()
        self.add_on_weight_dir = add_on_weight_dir.resolve()
        self.device = device
        self.model: Any | None = None
        self._scored_transcription: Any | None = None
        self._prepare_audio: Any | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if self.device.startswith('cuda') and self.device not in {'cuda', 'cuda:0'}:
            raise ContractError('Official WhiStress hard-codes bare CUDA in forward(); expose the selected physical GPU as logical cuda:0 instead of using cuda:N')
        import torch
        from transformers import WhisperConfig
        if self.device.startswith('cuda') and (not torch.cuda.is_available()):
            raise ContractError('CUDA device requested but torch.cuda.is_available() is false')
        if self.device == 'cpu' and torch.cuda.is_available():
            raise ContractError('Official WhiStress CPU execution requires CUDA_VISIBLE_DEVICES to be empty')
        if not (self.source_dir / 'whistress' / 'model' / 'model.py').is_file():
            raise FileNotFoundError(f'Invalid WhiStress source: {self.source_dir}')
        source_string = str(self.source_dir)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)
        model_module = importlib.import_module('whistress.model.model')
        utils_module = importlib.import_module('whistress.inference_client.utils')
        module_path = Path(model_module.__file__).resolve()
        if self.source_dir not in module_path.parents:
            raise ContractError(f'Imported WhiStress from unexpected path: {module_path}')
        model_class = model_module.WhiStress
        config = WhisperConfig.from_pretrained(self.base_model_dir, local_files_only=True)
        model = model_class(config, layer_for_head=9, whisper_backbone_name=str(self.base_model_dir))
        model.load_model(self.add_on_weight_dir, device=self.device)
        model.to(self.device)
        model.eval()
        tokenizer = model.processor.tokenizer
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
                raise ContractError('WhiStress tokenizer has neither pad nor EOS token')
            tokenizer.pad_token = tokenizer.eos_token
        model.processor.tokenizer.model_input_names = ['input_ids', 'attention_mask', 'labels_head']
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self._scored_transcription = utils_module.scored_transcription
        self._prepare_audio = utils_module.prepare_audio

    def _validate_transcription_length(self, transcription: str) -> None:
        assert self.model is not None
        untruncated = self.model.processor.tokenizer(transcription, return_tensors='pt', padding=False, truncation=False)['input_ids']
        if int(untruncated.shape[1]) > 30:
            raise TokenMapError(f'WhiStress fixed transcription would truncate: {untruncated.shape[1]} > 30')

    def predict_pairs(self, waveform: np.ndarray, sampling_rate: int, transcription: str) -> list[tuple[str, int]]:
        self.load()
        assert self.model is not None
        assert self._scored_transcription is not None
        self._validate_transcription_length(transcription)
        pairs = self._scored_transcription(audio={'sampling_rate': sampling_rate, 'array': waveform}, model=self.model, device=self.device, strip_words=True, transcription=transcription)
        return [(str(word), int(label)) for word, label in pairs]

    def predict_uncalibrated_word_scores(self, waveform: np.ndarray, sampling_rate: int, transcription: str) -> list[WordLogitSensitivity]:
        import torch
        self.load()
        assert self.model is not None
        assert self._prepare_audio is not None
        tokenizer = self.model.processor.tokenizer
        self._validate_transcription_length(transcription)
        input_ids = tokenizer(transcription, return_tensors='pt', padding='max_length', truncation=True, max_length=30)['input_ids']
        prepared_audio = self._prepare_audio({'sampling_rate': sampling_rate, 'array': waveform})
        input_features = self.model.processor.feature_extractor(prepared_audio, sampling_rate=16000, return_tensors='pt')['input_features']
        with torch.inference_mode():
            output = self.model(input_features=input_features.to(self.device), decoder_input_ids=input_ids.to(self.device))
            probabilities = torch.softmax(output.logits, dim=-1)
            binary = torch.argmax(probabilities, dim=-1)
            class1 = probabilities[..., 1]
            binary_shifted = torch.cat((binary[:, -1:], binary[:, :-1]), dim=1)
            class1_shifted = torch.cat((class1[:, -1:], class1[:, :-1]), dim=1)
        token_values = [int(value) for value in input_ids[0].tolist()]
        binary_values = [int(value) for value in binary_shifted[0].cpu().tolist()]
        score_values = [float(value) for value in class1_shifted[0].cpu().tolist()]
        return merge_shifted_token_outputs(token_values, binary_values, score_values, tokenizer)
