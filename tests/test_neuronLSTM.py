"""
.. module:: $Module_name
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""
from unittest import TestCase
from opthh.neuron import NeuronLSTM
import tensorflow as tf


class TestNeuronLSTM(TestCase):

    def test_build_graph(self):
        l = NeuronLSTM()
        i,r = l.build_graph(1)
        self.assertEqual(len(i.shape), 2)
        l.build_graph(4)
        l.build_graph()

    def test_init(self):
        l = NeuronLSTM()
        l.reset()
        with self.assertRaises(ReferenceError):
            l.hidden_init_state
        l.init(1)
        l.hidden_init_state

