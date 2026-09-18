import math
import torch
from torch import nn

def sample_frame_times(source, source_hz, destination_frames, destination_hz, *, nearest=False):
    if source.ndim != 2 or source.shape[-1] < 1 or source_hz <= 0 or (destination_hz <= 0) or (destination_frames < 1):
        raise ValueError('Invalid frame map')
    positions = torch.arange(destination_frames, device=source.device, dtype=torch.float64) * source_hz / destination_hz
    if positions[-1] > source.shape[-1] - 1 + source_hz / destination_hz + 1e-06:
        raise ValueError('Destination extends beyond source timeline')
    positions = positions.clamp(0, source.shape[-1] - 1)
    if nearest:
        return source[:, positions.round().long()]
    low, high = (positions.floor().long(), positions.ceil().long())
    fraction = (positions - low).to(source.dtype)
    return source[:, low] * (1 - fraction) + source[:, high] * fraction

class CtrlSpeechPhoneEmbedding(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pitch_embedding = nn.Embedding(128, channels)
        self.loudness_embedding = nn.Embedding(64, channels)
        self.duration_embedding = nn.Embedding(192, channels)

    def forward(self, pitch_bins, loudness_bins, phone_bounds, *, control_dropout=False):
        if pitch_bins.ndim != 1 or pitch_bins.shape != loudness_bins.shape:
            raise ValueError('Expected aligned one-dimensional frame codes')
        if phone_bounds.ndim != 2 or phone_bounds.shape[1] != 2 or len(phone_bounds) == 0:
            raise ValueError('Expected explicit [phone,2] bounds')
        if any((t.dtype != torch.long for t in (pitch_bins, loudness_bins, phone_bounds))):
            raise ValueError('Codes/bounds must be int64')
        if (phone_bounds < 0).any() or (phone_bounds[:, 1] < phone_bounds[:, 0]).any() or (phone_bounds[:, 1] > len(pitch_bins)).any():
            raise ValueError('Phone falls outside acoustic timeline')
        if (phone_bounds[:, 0] >= len(pitch_bins)).any():
            raise ValueError('Empty phone beyond acoustic timeline')
        duration = phone_bounds[:, 1] - phone_bounds[:, 0]
        if (duration > 191).any():
            raise ValueError('Phone exceeds original duration vocabulary; do not clip silently')
        p = self.pitch_embedding(pitch_bins)
        l = self.loudness_embedding(loudness_bins)
        pooled_p = torch.stack([p[s:e].mean(0) if e > s else p[s] for s, e in phone_bounds])
        pooled_l = torch.stack([l[s:e].mean(0) if e > s else l[s] for s, e in phone_bounds])
        branches = [pooled_p, pooled_l, self.duration_embedding(duration)]
        if self.training and control_dropout:
            branches = [x * (torch.rand((), device=x.device) < 0.5) for x in branches]
        return sum(branches)

class WordVoiceFrameModulation(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        if channels % 4:
            raise ValueError('WordVoice attribute width must be divisible by four')
        self.channels = channels
        self.bnd_embed = nn.Embedding(6, channels // 4)
        self.tone_embed = nn.Embedding(8, channels // 4)
        self.f0_embed = nn.Embedding(21, channels // 4)
        self.energy_embed = nn.Embedding(21, channels // 4)
        self.control_modulator = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 2 * channels))
        nn.init.zeros_(self.control_modulator[-1].weight)
        nn.init.zeros_(self.control_modulator[-1].bias)

    def forward(self, hidden, boundary, tone, f0, energy, duration_frames):
        if hidden.ndim != 2 or hidden.shape[-1] != self.channels:
            raise ValueError('Expected [Mel frame, channels] native hidden state')
        inputs = (boundary, tone, f0, energy, duration_frames)
        if any((x.ndim != 1 or len(x) != len(boundary) or x.dtype != torch.long for x in inputs)):
            raise ValueError('Word attributes and durations must be aligned int64 vectors')
        if (duration_frames < 0).any() or int(duration_frames.sum()) != len(hidden):
            raise ValueError('Explicit durations must cover the full destination timeline')
        attributes = torch.cat([self.bnd_embed(boundary), self.tone_embed(tone), self.f0_embed(f0), self.energy_embed(energy)], -1)
        position = torch.arange(len(attributes), device=hidden.device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.channels, 2, device=hidden.device, dtype=torch.float32) * (-math.log(10000.0) / self.channels))
        pe = torch.zeros_like(attributes)
        pe[:, 0::2], pe[:, 1::2] = (torch.sin(position * div), torch.cos(position * div))
        frame_attributes = torch.repeat_interleave(attributes + pe, duration_frames, dim=0)
        scale, shift = self.control_modulator(frame_attributes).chunk(2, dim=-1)
        return hidden * (1 + torch.tanh(scale)) + shift
