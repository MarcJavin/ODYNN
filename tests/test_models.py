from unittest import TestCase
from odynn.models import cfg_model
import numpy as np

p = cfg_model.NEURON_MODEL.default_params

models2 = cfg_model.models

class TestModels(TestCase):

    def test_init(self):
        for mod in models2:
            p = mod.default_params
            hh = mod(init_p=[p for _ in range(10)])
            self.assertEqual(len(hh._init_state), len(hh.default_init_state))
            self.assertEqual(hh.num, 10)
            self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),hh.num))
            self.assertIsInstance(hh._init_p, dict)
            self.assertEqual(list(hh._init_p.values())[0].shape, (10,))
            self.assertEqual(hh._init_p.keys(), p.keys())
            self.assertEqual(hh._init_p, hh._param)
            self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))
    
            hh = mod(init_p=[mod.get_random() for _ in range(13)])
            self.assertEqual(len(hh._init_state), len(hh.default_init_state))
            self.assertEqual(hh.num, 13)
            self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
            self.assertIsInstance(hh._init_p, dict)
            self.assertEqual(list(hh._init_p.values())[0].shape, (13,))
            self.assertEqual(hh._init_p.keys(), p.keys())
            self.assertEqual(hh._init_p, hh._param)
            self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))
    
            hh = mod(p)
            self.assertEqual(hh.num, 1)
            self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
            self.assertIsInstance(hh._init_p, dict)
            self.assertEqual(hh._init_p, p)
            self.assertEqual(hh._init_p, hh._param)
            self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))


    def test_plot_results(self):
        for mod in models2:
            hh = mod(mod.default_params)
            ts = np.arange(0, 100., 0.1)
            i = np.zeros(len(ts))
            res = np.ones((len(ts), len(mod.default_init_state)))
            hh.plot_results(ts, i, res, show=False)

            hh = mod(init_p=[mod.get_random() for _ in range(12)])
            ts = np.arange(0, 100., 0.1)
            i = np.zeros((len(ts), 12))
            res = np.ones((len(ts), len(mod.default_init_state), 12))
            hh.plot_results(ts, i, res, show=False)

    def test_step(self):
        for mod in models2:
            hh = mod(mod.default_params)
            i = 1.
            X = mod.default_init_state
            X2 = hh.step(X, i)
            self.assertEqual(X.shape, X2.shape)

            mod(init_p=[mod.get_random() for _ in range(7)])
            i = np.array([1. for _ in range(7)])
            X = np.stack([mod.default_init_state for _ in range(7)], axis=1)
            X2 = hh.step(X, i)
            self.assertEqual(X.shape, X2.shape)
