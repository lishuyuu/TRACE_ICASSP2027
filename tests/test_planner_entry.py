import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from trace_tts.trace_networks import RelationalPlanner
ROOT = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location('planner_entry', ROOT / 'scripts/train_planner.py')
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)

class PreparedInputTests(unittest.TestCase):

    def setUp(self):
        self.cfg = dict(native_dim=2, speaker_dim=1, width=4, layers=1, heads=1, feedforward_dim=4, dropout=0.0, position_base=10000.0, activation='gelu', norm_first=True, norm_eps=1e-05)
        self.omega = torch.tensor([1.0, 0.0])
        self.model = RelationalPlanner(omega=self.omega, **self.cfg)
        self.group = dict(split='train', native_word_states=torch.zeros(2, 3, 2), targets=torch.tensor([0, 1]), speaker_embeddings=torch.zeros(2, 1), valid_words=torch.ones(2, 3, dtype=torch.bool), oracle_coordinates=torch.zeros(2, 3, 2), observed=torch.ones(2, 3, 2, dtype=torch.bool))

    def test_valid_inputs_do_not_change_model(self):
        before = copy.deepcopy(self.model.state_dict())
        ENTRY.validate_groups([self.group], self.model)
        self.assertTrue(all((torch.equal(before[k], value) for k, value in self.model.state_dict().items())))

    def test_all_groups_checked_not_only_sampled_group(self):
        bad = dict(self.group, split='test')
        with self.assertRaisesRegex(ValueError, 'Group 1'):
            ENTRY.validate_groups([self.group, bad], self.model)

    def test_repeated_target_is_not_complete_group(self):
        with self.assertRaisesRegex(ValueError, 'distinct targets'):
            ENTRY.validate_groups([dict(self.group, targets=torch.tensor([0, 0]))], self.model)

    def test_wrong_oracle_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'oracle shape'):
            ENTRY.validate_groups([dict(self.group, oracle_coordinates=torch.zeros(2, 3, 1))], self.model)

    def test_no_observation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no observed supervision'):
            ENTRY.validate_groups([dict(self.group, observed=torch.zeros(2, 3, 2, dtype=torch.bool))], self.model)

    def test_nonfinite_hyperparameters_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for position in range(3):
                settings = [0.001, 0.0, 1.0]
                settings[position] = value
                with self.subTest(value=value, position=position), self.assertRaises(ValueError):
                    ENTRY.validate_settings(1, *settings)

    def test_nonfinite_feature_rejected(self):
        features = self.group['native_word_states'].clone()
        features[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            ENTRY.validate_groups([dict(self.group, native_word_states=features)], self.model)

    def test_validate_cli_saves_no_weights(self):
        with tempfile.TemporaryDirectory(prefix='trace-input-check-') as directory:
            root = Path(directory)
            groups, config = (root / 'groups.pt', root / 'config.json')
            torch.save({'groups': [self.group], 'omega': self.omega}, groups)
            config.write_text(json.dumps({'planner': self.cfg}), encoding='utf-8')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_planner.py'), '--groups', str(groups), '--config', str(config), '--seed', '2704', '--steps', '1', '--lr', '0.001', '--weight-decay', '0', '--huber-delta', '1', '--validate-only'], capture_output=True, text=True, timeout=60, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('no training performed', result.stdout)
            self.assertEqual({p.name for p in root.iterdir()}, {'groups.pt', 'config.json'})
if __name__ == '__main__':
    unittest.main()
