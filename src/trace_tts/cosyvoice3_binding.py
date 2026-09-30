"""CosyVoice3 host binding for the shared TRACE inference path.

The upstream CosyVoice3 Python API exposes synthesis as completed waveforms.
TRACE also needs the host's native acoustic state, word alignment, flow-field
evaluation, sampler, and decoder. ``CosyVoice3Runtime`` is the narrow bridge to
those checkpoint- and revision-specific internals; this class implements the
``BackboneAdapter`` contract and handles host validation/freezing consistently.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import torch
from torch import Tensor, nn

from .backend_contract import NativeState, TransportedCondition


class CosyVoice3Runtime(Protocol):
    """Operations supplied by the selected CosyVoice3 host revision.

    Implementations should use the loaded host's native frontend and flow
    components. ``native`` must honor ``seed`` and return word states and timing
    from that same native draw. ``integrate`` must preserve the native sampling
    context carried in ``NativeState.context`` / ``TransportedCondition.context``.
    """

    def native(
        self, host: Any, text: str, prompt: Any, seed: int
    ) -> NativeState: ...

    def transport(
        self, host: Any, native: NativeState, word_durations: Tensor
    ) -> TransportedCondition: ...

    def velocity(
        self,
        host: Any,
        state: Tensor,
        time: Tensor,
        prepared: TransportedCondition,
    ) -> Tensor: ...

    def integrate(
        self,
        host: Any,
        native: NativeState,
        prepared: TransportedCondition,
        field: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor: ...

    def decode(
        self, host: Any, state: Tensor, prepared: TransportedCondition
    ) -> Any: ...


class CosyVoice3Backend:
    """Adapt a loaded upstream ``CosyVoice3`` host to TRACE's backend contract.

    Load the host with the selected CosyVoice3 release (for example via its
    ``AutoModel(model_dir=...)``), then provide a runtime implementing the five
    operations above for that exact upstream revision and checkpoint.
    """

    def __init__(self, host: Any, runtime: CosyVoice3Runtime) -> None:
        required = ("model", "frontend", "sample_rate")
        missing = [name for name in required if not hasattr(host, name)]
        if missing:
            raise TypeError(
                "host must be a loaded CosyVoice3 object; missing "
                + ", ".join(missing)
            )
        if not all(
            callable(getattr(runtime, name, None))
            for name in ("native", "transport", "velocity", "integrate", "decode")
        ):
            raise TypeError("runtime must implement all CosyVoice3Runtime operations")
        self.host = host
        self.runtime = runtime

    def freeze(self) -> None:
        """Freeze and switch the acoustic model components to evaluation mode."""
        model = self.host.model
        modules: list[nn.Module] = []
        if isinstance(model, nn.Module):
            modules.append(model)
        else:
            for name in ("llm", "flow", "hift"):
                component = getattr(model, name, None)
                if isinstance(component, nn.Module):
                    modules.append(component)
        if not modules:
            raise TypeError(
                "CosyVoice3 host exposes no torch modules among model, llm, flow, hift"
            )
        for module in modules:
            module.eval()
            module.requires_grad_(False)

    def native(self, text: str, prompt: Any, seed: int) -> NativeState:
        return self.runtime.native(self.host, text, prompt, seed)

    def transport(
        self, native: NativeState, word_durations: Tensor
    ) -> TransportedCondition:
        return self.runtime.transport(self.host, native, word_durations)

    def velocity(
        self, state: Tensor, time: Tensor, prepared: TransportedCondition
    ) -> Tensor:
        return self.runtime.velocity(self.host, state, time, prepared)

    def integrate(
        self,
        native: NativeState,
        prepared: TransportedCondition,
        field: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        return self.runtime.integrate(self.host, native, prepared, field)

    def decode(self, state: Tensor, prepared: TransportedCondition) -> Any:
        return self.runtime.decode(self.host, state, prepared)
