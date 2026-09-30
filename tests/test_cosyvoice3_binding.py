import unittest

import torch

from trace_tts.cosyvoice3_binding import CosyVoice3Backend


class _HostModel:
    def __init__(self):
        self.llm = torch.nn.Linear(2, 2)
        self.flow = torch.nn.Linear(2, 2)
        self.hift = torch.nn.Linear(2, 2)


class _Host:
    def __init__(self):
        self.model = _HostModel()
        self.frontend = object()
        self.sample_rate = 24000


class _Runtime:
    def native(self, host, text, prompt, seed):
        return (host, text, prompt, seed)

    def transport(self, host, native, word_durations):
        return (host, native, word_durations)

    def velocity(self, host, state, time, prepared):
        return (host, state, time, prepared)

    def integrate(self, host, native, prepared, field):
        return (host, native, prepared, field)

    def decode(self, host, state, prepared):
        return (host, state, prepared)


class CosyVoice3BindingTests(unittest.TestCase):
    def test_freeze_freezes_cosvoice_components(self):
        host = _Host()
        backend = CosyVoice3Backend(host, _Runtime())

        backend.freeze()

        for name in ("llm", "flow", "hift"):
            module = getattr(host.model, name)
            self.assertFalse(module.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))

    def test_native_forwards_seed_and_inputs(self):
        host = _Host()
        backend = CosyVoice3Backend(host, _Runtime())

        result = backend.native("text", "prompt.wav", 2703)

        self.assertEqual(result, (host, "text", "prompt.wav", 2703))

    def test_rejects_an_object_without_cosyvoice_host_fields(self):
        with self.assertRaisesRegex(TypeError, "loaded CosyVoice3"):
            CosyVoice3Backend(object(), _Runtime())


if __name__ == "__main__":
    unittest.main()
