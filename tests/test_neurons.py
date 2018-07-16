"""
.. module:: $Module_name
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""
from unittest import TestCase
from opthh import neuron as nr
import numpy as np


class TestNeurons(TestCase):


    def test_init(self):
        dt = 0.5
        p = nr.MODEL.get_random()
        neurons = nr.Neurons(
            [nr.BioNeuronTf(init_p=[p for _ in range(2)], dt=dt), nr.NeuronLSTM(dt=dt)])
        self.assertEqual(neurons.num, 3)
        self.assertEqual(neurons.init_state.all(), np.stack([nr.MODEL.default_init_state for _ in range(3)], axis=1).all())

        with self.assertRaises(AttributeError):
            neurons = nr.Neurons(
            [nr.BioNeuronTf(init_p=[p for _ in range(2)], dt=0.1), nr.NeuronLSTM(dt=0.2)])
