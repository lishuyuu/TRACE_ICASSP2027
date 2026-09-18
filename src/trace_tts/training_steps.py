import torch
from .trace_core import sample_alternatives, gcra_loss
from .signed_refinement import signed_renderer_loss

def adapter_fit_step(adapter, optimizer, *, state, time, condition, commands, target, valid_frames, generator, lambda_swap, margin, gamma):
    indices = sample_alternatives(state.shape[0], generator=generator).to(commands.device)
    optimizer.zero_grad(set_to_none=True)
    correct, swapped = adapter.forward_pair(state, time, condition, commands, commands[indices], valid_frames)
    value = gcra_loss(correct, swapped, target, valid_frames, commands, commands[indices], time=time, lambda_swap=lambda_swap, margin=margin, gamma=gamma)
    value.loss.backward()
    optimizer.step()
    return value

def adapter_refine_step(adapter, optimizer, *, state, time, condition, commands, target, endpoints, valid_frames, lambda_cf, lambda_p, margin, cosine_epsilon):
    optimizer.zero_grad(set_to_none=True)
    value = signed_renderer_loss(adapter, state, time, condition, commands, target, endpoints, valid_frames, lambda_cf=lambda_cf, lambda_p=lambda_p, margin=margin, cosine_epsilon=cosine_epsilon)
    loss = value.loss
    loss.backward()
    optimizer.step()
    return (loss.detach(), value)
