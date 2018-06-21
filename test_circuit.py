from unittest import TestCase
import params
from Circuit import Circuit
import numpy as np
from Neuron import Neuron_tf


class TestCircuit(TestCase):
    neuron = Neuron_tf([params.DEFAULT for _ in range(5)])
    conns = {(0, 4): params.SYNAPSE,
             (1, 4): params.SYNAPSE,
             (2, 4): params.SYNAPSE,
             (3, 2): params.SYNAPSE,
             }
    circ = Circuit(conns, neuron)

    conns2 = [{(0, 4): params.SYNAPSE,
             (1, 4): params.SYNAPSE,
             (2, 4): params.SYNAPSE,
             (3, 2): params.SYNAPSE,
             } for _ in range(7)]
    circ2 = Circuit(conns2, neuron)

    def test_init(self):
        neuron_bad = Neuron_tf([params.DEFAULT for _ in range(3)])

        self.circ = Circuit(self.conns, self.neuron)
        self.assertEqual(self.circ.num, 1)
        self.assertEqual(self.circ.pres.all(), np.array([0, 1, 2, 3]).all())
        self.assertEqual(self.circ.posts.all(), np.array([4, 4, 4, 2]).all())
        self.assertIsInstance(self.circ.param, dict)
        self.assertEqual(self.circ.param.keys(), params.SYNAPSE.keys())
        self.assertEqual(self.circ.neurons.num, 5)

        self.assertEqual(self.circ2.num, 7)
        self.assertEqual(self.circ2.pres.all(), np.array([0, 1, 2, 3]).all())
        self.assertEqual(self.circ2.posts.all(), np.array([4, 4, 4, 2]).all())
        self.assertIsInstance(self.circ2.param, dict)
        self.assertEqual(self.circ2.param.keys(), params.SYNAPSE.keys())
        self.assertEqual(self.circ2.neurons.num, 5)
        self.assertEqual(self.circ2.neurons.init_state.shape[-1], self.circ2.num)

        with self.assertRaises(AssertionError):
            cbad = Circuit(self.conns, neuron_bad)
        with self.assertRaises(AssertionError):
            cbad = Circuit(self.conns2, neuron_bad)



    def test_syn_curr(self):
        vp = [0,3,0,1.1,4]
        vpo = [1., 2., 0, 0, 4]
        with self.assertRaises(ValueError):
            self.circ.syn_curr(vp, vpo[:-1])
        with self.assertRaises(ValueError):
            self.circ.syn_curr(vp[-2], vpo[:-2])
        with self.assertRaises(ValueError):
            self.circ.syn_curr(vp, vpo)
        i = self.circ.syn_curr(vp[:-1], vpo[:-1])
        self.assertEqual(len(i), 4)

        vp = np.stack([vp[:-1] for _ in range(7)], axis=1)
        vpo = np.stack([vpo[:-1] for _ in range(7)], axis=1)
        i = self.circ2.syn_curr(vp, vpo)
        self.assertEqual(i.shape, (4,7))

        vp = np.stack([vp for _ in range(10)], axis=0)
        vpo = np.stack([vpo for _ in range(10)], axis=0)
        i = self.circ2.syn_curr(vp, vpo)
        self.assertEqual(i.shape, (10, 4, 7))

    def test_step(self):
        pass



