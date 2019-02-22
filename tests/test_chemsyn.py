"""
.. module::
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from unittest import TestCase
import numpy as np
import torch
from odynn.models.chemsyn import ChemSyn

class TestChemSyn(TestCase):


    def test_step(self):
        s = ChemSyn(np.array([0]), np.array([1]))
        v = np.ones((1, 1, 2, 1))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = ChemSyn(np.array([0, 1, 2, 3]), np.array([1, 2, 1, 1]))
        v = np.ones((1, 1, 4, 1))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = ChemSyn(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]))
        v = np.ones((1, 1, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = ChemSyn(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]))
        v = np.ones((1, 5, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = ChemSyn(np.array([0]), np.array([1]), tensors=True)
        v = torch.ones((1, 1, 2, 1))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = ChemSyn(np.array([0, 1, 2, 3]), np.array([1, 2, 1, 1]), tensors=True)
        v = torch.ones((1, 1, 4, 1))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = ChemSyn(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]), tensors=True)
        v = torch.ones((1, 1, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = ChemSyn(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]), tensors=True)
        v = torch.ones((1, 5, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = ChemSyn(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]), init_p=ChemSyn.get_random(4,3), tensors=True)
        v = torch.ones((1, 5, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

