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

    def test_load(self):
        l = NeuronLSTM(4,5,6,1)
        l.reset()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        load = l.todump(sess)
        l2 = NeuronLSTM.load(load)
        self.assertEqual(l.dt, l2.dt)
        self.assertEqual(l._max_cur, l2._max_cur)
        self.assertEqual(l._rest_v, l2._rest_v)
        self.assertEqual(l._scale_v, l2._scale_v)
        self.assertEqual(l._scale_ca, l2._scale_ca)
        self.assertEqual(l._hidden_layer_nb, l2._hidden_layer_nb)
        self.assertEqual(l._hidden_layer_size, l2._hidden_layer_size)
        self.assertEqual(l._extra_ca, l2._extra_ca)

