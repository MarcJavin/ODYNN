"""
.. self.clule::
    :synopsis: self.clule doing stuff...

.. self.cluleauthor:: Marc Javin
"""

from unittest import TestCase
import numpy as np
import torch
from odynn.models import model


class testModel(TestCase):


    class cl(model.Model):

        _random_bounds = {'a': [10, 42], 'c': [2.5, 10.2]}

        def step(self,a,b):
            pass

    cl.default_params = {'a': 20, 'c': 10}
    cl.default_init_state = np.array([4])

    mod = cl()

    def test_get_random(self):
        p = self.cl.get_random(24,5)
        self.assertEqual(p['a'].shape, (24,5))
        self.assertTrue((p['a'] > self.cl._random_bounds['a'][0]).all())
        self.assertTrue((p['a'] < self.cl._random_bounds['a'][1]).all())

    def test_num(self):
        self.mod._num = 5
        self.assertEqual(5, self.mod.num)

    def test_init_state(self):
        a = 54
        self.mod._init_state = a
        self.assertEqual(self.mod.init_state, 54)

        b = 'fg'
        self.cl.default_init_state = b
        inst = self.cl()
        self.assertEqual(inst.init_state, b)

    def test_init(self):
        p = {'a': 20, 'b':0.1}
        hh = self.cl(init_p={k: np.ones((10, 4)) for k in p.keys()})
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh.parallel, 4)
        # self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh._init_p, dict)
        self.assertEqual(list(hh._init_p.values())[0].shape, (10,4))
        self.assertEqual(hh._init_p.keys(), p.keys())
        self.assertEqual(hh._init_p, hh._param)
        self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))

        hh = self.cl(init_p={k: np.ones((11, 5)) for k in p.keys()}, tensors=True)
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 11)
        self.assertEqual(hh.parallel, 5)
        # self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh._init_p, dict)
        self.assertEqual(list(hh._init_p.values())[0].shape, (11,5))
        self.assertEqual(hh._init_p.keys(), p.keys())
        torchp = {k: torch.Tensor(v) for k, v in hh._init_p.items()}
        for k in hh._init_p.keys():
            self.assertTrue(torch.all(torch.eq(torchp[k], hh._param[k])))
        self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))


    def test_create_random(self):
        hh = self.cl.create_random(5,6, tensors=False)
        self.assertEqual(hh.num, 5)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
        self.assertIsInstance(hh._init_p, dict)
        self.assertEqual(hh._init_p, hh._param)
        self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))

        hh = self.cl.create_random(9, 3)
        self.assertEqual(hh.num, 9)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
        self.assertIsInstance(hh._init_p, dict)
        torchp = {k: torch.Tensor([v]) for k, v in hh._init_p.items()}
        for k in hh._init_p.keys():
            self.assertTrue(torch.all(torch.eq(torchp[k], hh._param[k])))
        self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))

    def test_create_default(self):
        hh = self.cl.create_default(5,6, tensors=False)
        self.assertEqual(hh.num, 5)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
        self.assertIsInstance(hh._init_p, dict)
        self.assertEqual(hh._init_p, hh._param)
        self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))

        hh = self.cl.create_random(9, 3)
        self.assertEqual(hh.num, 9)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
        self.assertIsInstance(hh._init_p, dict)
        torchp = {k: torch.Tensor([v]) for k, v in hh._init_p.items()}
        for k in hh._init_p.keys():
            self.assertTrue(torch.all(torch.eq(torchp[k], hh._param[k])))
        self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))
        
        
class testNeuronModel(TestCase):

    class cl(model.NeuronModel):

        def step(self,V,i):
            return i


    def test_calculate(self):
        n = self.cl()
        i = np.array([1,2,3,9,5,6])
        res = n.calculate(i)
        self.assertEqual(i.all(), res.all())

