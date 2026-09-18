import argparse
import json
import math
import random
from pathlib import Path
import torch
from trace_tts.trace_networks import RelationalPlanner, observed_huber_loss

def validate_settings(steps, lr, weight_decay, huber_delta):
    if type(steps) is not int or steps < 1 or any((isinstance(v, bool) or not math.isfinite(v) for v in (lr, weight_decay, huber_delta))) or (lr <= 0) or (weight_decay < 0) or (huber_delta <= 0):
        raise ValueError('Training settings must be finite; steps/lr/delta positive and decay nonnegative')

def validate_groups(groups, model):
    if not isinstance(groups, list) or not groups:
        raise ValueError('Supply a nonempty list of TRAIN complete groups')
    fields = {'split', 'native_word_states', 'targets', 'speaker_embeddings', 'valid_words', 'oracle_coordinates', 'observed'}
    with torch.no_grad():
        for index, group in enumerate(groups):
            if not isinstance(group, dict) or not fields.issubset(group):
                raise ValueError(f'Group {index}: missing required tensor fields')
            if group['split'] != 'train':
                raise ValueError(f'Group {index}: only TRAIN groups are allowed')
            _, _, valid, counts, _ = model._validated_inputs(group['native_word_states'], group['speaker_embeddings'], group['valid_words'])
            targets = group['targets']
            if not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64 or targets.shape != (valid.shape[0],) or (targets.device != valid.device):
                raise ValueError(f'Group {index}: one int64 same-device target per endpoint required')
            if targets.numel() < 2 or targets.unique().numel() != targets.numel() or (not bool(((targets >= 0) & (targets < counts)).all())):
                raise ValueError(f'Group {index}: need multiple distinct targets on actual words')
            if not torch.equal(valid, valid[:1].expand_as(valid)):
                raise ValueError(f'Group {index}: matched alternatives must share a word mask')
            oracle = group['oracle_coordinates']
            expected = (*valid.shape, model.omega.numel())
            if not isinstance(oracle, torch.Tensor) or tuple(oracle.shape) != expected or oracle.device != model.omega.device or (oracle.dtype != model.omega.dtype):
                raise ValueError(f'Group {index}: oracle shape/device/dtype differs from planner')
            observed_huber_loss(oracle, oracle, group['observed'], valid, delta=1.0)

def main():
    p = argparse.ArgumentParser(description='Train a relational planner from complete TRAIN groups')
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=[2703, 2704, 2705], required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--weight-decay', type=float, required=True)
    p.add_argument('--huber-delta', type=float, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path)
    p.add_argument('--validate-only', action='store_true', help='Check all supplied tensors and settings without optimizer updates or saved weights')
    a = p.parse_args()
    validate_settings(a.steps, a.lr, a.weight_decay, a.huber_delta)
    if not a.validate_only and a.output is None:
        p.error('--output is required unless --validate-only is used')
    if not a.validate_only and a.output.exists():
        raise FileExistsError(a.output)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    data = torch.load(a.groups, map_location=a.device, weights_only=True)
    groups = data['groups']
    model = RelationalPlanner(omega=data['omega'], **cfg['planner']).to(a.device)
    validate_groups(groups, model)
    if a.validate_only:
        print(f'Validated {len(groups)} supplied TRAIN groups; no training performed.')
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    for _ in range(a.steps):
        g = groups[rng.randrange(len(groups))]
        optimizer.zero_grad(set_to_none=True)
        out = model(g['native_word_states'], g['targets'], g['speaker_embeddings'], g['valid_words'])
        loss = observed_huber_loss(out.coordinates, g['oracle_coordinates'], g['observed'], g['valid_words'], delta=a.huber_delta).loss
        loss.backward()
        if any((parameter.grad is not None and (not bool(torch.isfinite(parameter.grad).all())) for parameter in model.parameters())):
            raise FloatingPointError('Non-finite planner gradient; checkpoint not saved')
        optimizer.step()
    if any((not bool(torch.isfinite(parameter).all()) for parameter in model.parameters())):
        raise FloatingPointError('Non-finite planner parameter; checkpoint not saved')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {'format_version': 1, 'planner': model.state_dict(), 'model_config': cfg['planner'], 'seed': a.seed, 'omega': data['omega'], 'steps': a.steps, 'training_config': {'optimizer': 'AdamW', 'lr': a.lr, 'weight_decay': a.weight_decay, 'huber_delta': a.huber_delta, 'sampling': 'uniform_complete_groups'}, 'group_count': len(groups), 'torch_version': str(torch.__version__)}
    with a.output.open('xb') as stream:
        torch.save(checkpoint, stream)
if __name__ == '__main__':
    main()
