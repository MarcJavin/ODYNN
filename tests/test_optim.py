"""
.. module:: 
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from unittest import TestCase
from odin import optim, neuron, circuit
import numpy as np


class TestOptim(TestCase):


    def test_plot_loss_rate(self):
        loss = np.array([4., 3. ,2. ,1.])
        rates = np.array([3., 2.5, 2., 1.5])
        optim.plot_loss_rate(loss, rates, save=False, show=False)
        loss_test = np.array([3., 3., 2., 1.9])
        optim.plot_loss_rate(loss, rates, loss_test, save=False, show=False)
        loss = np.stack([loss for _ in range(7)], axis=-1)
        optim.plot_loss_rate(loss, rates, save=False, show=False)
        loss_test = np.stack([loss_test for _ in range(7)], axis=-1)
        optim.plot_loss_rate(loss, rates, loss_test, save=False, show=False)

    def test_init(self):

        class opt(optim.Optimizer):
            def _build_loss(self, w):
                pass

        nr = neuron.BioNeuronTf(n_rand=3)
        op = opt(nr)
        self.assertEqual(op._parallel, nr.num)
        self.assertEqual(nr.num, 3)

        c = circuit.CircuitTf.create_random(2, syn_keys={(0,1):True}, n_rand=7)
        op = opt(c)
        self.assertEqual(op._parallel, c.num)
        self.assertEqual(c.num, 7)