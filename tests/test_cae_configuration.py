import unittest

import numpy as np

from trace_tts.cae_reference import fit_mp_cae_model, apply_mp_cae_target_mu


class CAEConfigurationTests(unittest.TestCase):
    def test_explicit_geometry_and_hook_are_used(self):
        features = np.array([[-1., -2.], [-1., 2.], [1., -2.], [1., 2.]])
        model = fit_mp_cae_model(features, features.copy(),
                                 fit_ids=['synthetic_a', 'synthetic_b', 'synthetic_c', 'synthetic_d'],
                                 group_ids=['g1', 'g1', 'g2', 'g2'],
                                 pca_dim=2, ridge=.1, trust_radius=2., hook='synthetic_condition')
        self.assertEqual(model.pca_dim, 2)
        self.assertEqual(model.trust_radius, 2.)
        self.assertEqual(model.hook, 'synthetic_condition')
        source = np.array([[1., 2., 3.], [2., 1., 2.]])
        edited, _ = apply_mp_cae_target_mu(source, np.arange(3), [1], model,
                                          pitch_semitones=.5, energy_db=.25, trust_radius=2.)
        np.testing.assert_array_equal(edited[:, [0, 2]], source[:, [0, 2]])
        self.assertFalse(np.array_equal(edited[:, 1], source[:, 1]))


if __name__ == '__main__':
    unittest.main()
