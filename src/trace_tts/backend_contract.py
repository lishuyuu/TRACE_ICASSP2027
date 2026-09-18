from dataclasses import dataclass
from typing import Any, Callable, Protocol
from torch import Tensor

@dataclass(frozen=True)
class NativeState:
    waveform: Any
    word_states: Tensor
    speaker: Tensor
    word_durations: Tensor
    boundary_strengths: Tensor
    context: Any

@dataclass(frozen=True)
class TransportedCondition:
    flow_condition: Any
    adapter_condition: Any
    word_index: Tensor
    nucleus_phase: Tensor
    voicing_gate: Tensor
    valid_frames: Tensor
    context: Any

class BackboneAdapter(Protocol):

    def freeze(self) -> None:
        ...

    def native(self, text: str, prompt: Any, seed: int) -> NativeState:
        ...

    def transport(self, native: NativeState, word_durations: Tensor) -> TransportedCondition:
        ...

    def velocity(self, state: Tensor, time: Tensor, prepared: TransportedCondition) -> Tensor:
        ...

    def integrate(self, native: NativeState, prepared: TransportedCondition, field: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        ...

    def decode(self, state: Tensor, prepared: TransportedCondition) -> Any:
        ...
