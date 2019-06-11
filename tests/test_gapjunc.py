"""
.. module::
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from unittest import TestCase
import numpy as np
import torch
from odynn.models.gapjunc import GapJunction

class TestGapJunction(TestCase):


    def test_step(self):
        s = GapJunction(np.array([0]), np.array([1]))
        v = np.ones((1, 1, 2, 1))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = GapJunction(np.array([0, 1, 2, 3]), np.array([1, 2, 1, 1]))
        v = np.ones((1, 1, 4, 1))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = GapJunction(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]))
        v = np.ones((1, 1, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = GapJunction(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]))
        v = np.ones((1, 5, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = GapJunction(np.array([0]), np.array([1]), tensors=True)
        v = torch.ones((1, 1, 2, 1))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = GapJunction(np.array([0, 1, 2, 3]), np.array([1, 2, 1, 1]), tensors=True)
        v = torch.ones((1, 1, 4, 1))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = GapJunction(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]), tensors=True)
        v = torch.ones((1, 1, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = GapJunction(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]), tensors=True)
        v = torch.ones((1, 5, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

        s = GapJunction(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 1]), init_p=GapJunction.get_random(4,3), tensors=True)
        v = torch.ones((1, 5, 4, 3))
        c = s.step(v)
        self.assertEqual(c.shape, v.shape[1:])

