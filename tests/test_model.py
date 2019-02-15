"""
.. self.clule::
    :synopsis: self.clule doing stuff...

.. self.cluleauthor:: Marc Javin
"""

from unittest import TestCase
import numpy as np
from odynn.models import model


class testNeuron(TestCase):
    class cl(model.Neuron):

        def step(self, a, b):
            pass

        def calculate(self, i):
            pass

    mod = cl()

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


class testBioNeuron(TestCase):
    class cl(model.BioNeuron):

        def step(self, X, i):
            return i

    cl.default_params = {'a': 20, 'c': 10}
    cl.default_init_state = np.array([4])

    def test_init(self):
        p = {'a': 20, 'b': 0.1}
        hh = self.cl(init_p=[p for _ in range(10)])
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh._init_p, dict)
        self.assertEqual(list(hh._init_p.values())[0].shape, (10,))
        self.assertEqual(hh._init_p.keys(), p.keys())
        self.assertEqual(hh._init_p, hh._param)
        self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))

        hh = self.cl(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
        self.assertIsInstance(hh._init_p, dict)
        self.assertEqual(hh._init_p, p)
        self.assertEqual(hh._init_p, hh._param)
        self.assertEqual(hh.parameter_names, list(hh.default_params.keys()))

    def test_calculate(self):
        n = self.cl()
        i = np.array([1, 2, 3, 9, 5, 6])
        res = n.calculate(i)
        self.assertEqual(i.all(), res.all())

