from unittest import TestCase
from context import odin
import odynn.circuit
from odynn.circuit import CircuitTf, Circuit
import numpy as np
from odynn.neuron import BioNeuronTf, PyBioNeuron
from odynn import utils
import pickle



class TestCircuit(TestCase):


    dir = utils.set_dir('unittest')
    neuron = BioNeuronTf([PyBioNeuron.default_params for _ in range(5)])
    conns = {(0, 4): odynn.circuit.SYNAPSE,
             (1, 4): odynn.circuit.SYNAPSE,
             (2, 4): odynn.circuit.SYNAPSE,
             (3, 2): odynn.circuit.SYNAPSE,
             }
    circ = Circuit(neuron, conns)

    conns2 = [{(0, 4): odynn.circuit.SYNAPSE,
             (1, 4): odynn.circuit.SYNAPSE,
             (2, 4): odynn.circuit.SYNAPSE,
             (3, 2): odynn.circuit.SYNAPSE,
             } for _ in range(7)]
    circ2 = Circuit(neuron, conns2)

    def test_init(self):
        neuron_bad = BioNeuronTf([PyBioNeuron.default_params for _ in range(3)])

        self.assertEqual(self.circ.num, 1)
        self.assertEqual(self.circ._pres.all(), np.array([0, 1, 2, 3]).all())
        self.assertEqual(self.circ._posts.all(), np.array([4, 4, 4, 2]).all())
        self.assertIsInstance(self.circ._param, dict)
        self.assertEqual(self.circ._neurons.num, 5)
        self.assertEqual(self.circ._neurons.init_state.all(), self.circ.init_state.all())
        self.assertEqual(self.circ.n_gap, 0)
        self.assertEqual(self.circ.n_synapse, 4)

        self.assertEqual(self.circ2.num, 7)
        self.assertEqual(self.circ2._pres.all(), np.array([0, 1, 2, 3]).all())
        self.assertEqual(self.circ2._posts.all(), np.array([4, 4, 4, 2]).all())
        self.assertIsInstance(self.circ2._param, dict)
        self.assertEqual(self.circ2._neurons.num, 5)
        self.assertEqual(self.circ2._neurons.init_state.shape[-1], self.circ2.num)
        self.assertEqual(self.circ2._neurons.init_state.all(), self.circ2.init_state.all())
        self.assertEqual(self.circ2.n_gap, 0)
        self.assertEqual(self.circ2.n_synapse, 4)

        with self.assertRaises(AttributeError):
            cbad = Circuit(neuron_bad, self.conns)
        with self.assertRaises(AttributeError):
            cbad = Circuit(neuron_bad, self.conns2)
        with self.assertRaises(AttributeError):
            cbad = Circuit(neuron_bad, self.conns2, gaps={'G_gap':7})
        with self.assertRaises(AttributeError):
            cbad = Circuit(neuron_bad, gaps=[{'G_gap':7} for _ in range(2)], synapses={(0, 4): odynn.circuit.SYNAPSE})

        Circuit(neurons=self.neuron, gaps={k: {'G_gap':0.5} for k in self.conns.keys()})

    def test_pickle(self):
        neuron = BioNeuronTf([PyBioNeuron.default_params for _ in range(5)])
        conns = {(0, 4): odynn.circuit.SYNAPSE,
                 (1, 4): odynn.circuit.SYNAPSE,
                 (2, 4): odynn.circuit.SYNAPSE,
                 (3, 2): odynn.circuit.SYNAPSE,
                 }
        c = CircuitTf(neuron, conns)
        with open(self.dir + 'yeee', 'wb') as f:
            pickle.dump(c, f)
        with open(self.dir + 'yeee', 'rb') as f:
            circ = pickle.load(f)
        self.assertEqual(c.num, circ.num)
        self.assertEqual(circ._pres.all(), c._pres.all())
        self.assertEqual(c._posts.all(), circ._posts.all())
        self.assertEqual(circ._neurons.num, c._neurons.num)
        self.assertEqual(circ._neurons.init_state.all(), c._neurons.init_state.all())
        self.assertEqual(c.n_gap, circ.n_gap)
        self.assertEqual(c.n_synapse, circ.n_synapse)
        for k in c.gaps.keys():
            self.assertEqual(c.gaps[k].all(), circ.gaps[k].all())
        for k in c.init_params.keys():
            self.assertEqual(c.init_params[k].all(), circ.init_params[k].all())

    def test_create_random(self):
        c = CircuitTf.create_random(n_neuron=3, syn_keys={(0, 1):True, (1, 2):False}, n_rand=4)
        self.assertEqual(c.num, 4)
        self.assertEqual(c.init_params['G'].shape[0], 2)
        self.assertEqual(c.init_params['G'].shape[-1], 4)
        self.assertEqual(c._pres.all(), np.array([0, 1]).all())
        self.assertEqual(c._posts.all(), np.array([1, 2]).all())

    def test_gap(self):

        circ = Circuit(self.neuron, self.conns, gaps={(0, 1): odynn.circuit.GAP})
        self.assertEqual(circ.n_gap, 1)
        self.assertEqual(circ.n_synapse, 4)
        circ = Circuit(self.neuron, {},
                       gaps={(0, 1): odynn.circuit.GAP, (2, 3): odynn.circuit.GAP, (0, 4): odynn.circuit.GAP})
        self.assertEqual(circ.n_gap, 3)
        self.assertEqual(circ.n_synapse, 0)



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

class TestCircuitTf(TestCase):
    conns = {(0, 4): odynn.circuit.SYNAPSE,
             (1, 4): odynn.circuit.SYNAPSE,
             (2, 4): odynn.circuit.SYNAPSE,
             (3, 2): odynn.circuit.SYNAPSE,
             }
    gaps = {(0, 3): odynn.circuit.GAP}

    conns2 = [{(0, 4): odynn.circuit.SYNAPSE,
               (1, 4): odynn.circuit.SYNAPSE,
               (2, 4): odynn.circuit.SYNAPSE,
               (3, 2): odynn.circuit.SYNAPSE,
               } for _ in range(4)]
    gaps2 = [{(0, 3): odynn.circuit.GAP} for _ in range(4)]
    neuron = BioNeuronTf([PyBioNeuron.default_params for _ in range(5)])

    def test_calculate(self):
        print('#1 batch nope')
        c = CircuitTf(self.neuron, self.conns, self.gaps)
        print('#2 batches')
        i = np.ones((7,2,5))
        st = c.calculate(i)
        self.assertEqual(st.shape[0], 7)
        self.assertEqual(st.shape[1], len(PyBioNeuron.default_init_state))
        self.assertEqual(st.shape[2], 2)
        self.assertEqual(st.shape[3], 5)

        print('#1 batch, 4 models nope')
        c = CircuitTf(self.neuron, self.conns2, self.gaps2)
        print('#2 batches, 4 models')
        i = np.ones((7, 2, 5, 4))
        st = c.calculate(i)
        self.assertEqual(st.shape[0], 7)
        self.assertEqual(st.shape[1], len(PyBioNeuron.default_init_state))
        self.assertEqual(st.shape[2], 2)
        self.assertEqual(st.shape[3], 5)
        self.assertEqual(st.shape[-1], 4)

class TestCircuitFix(TestCase):

    dir = utils.set_dir('unittest')
    conns = {(0, 4): odynn.circuit.SYNAPSE,
             (1, 4): odynn.circuit.SYNAPSE,
             (2, 4): odynn.circuit.SYNAPSE,
             (3, 2): odynn.circuit.SYNAPSE,
             }
    gaps = {(0,3): odynn.circuit.GAP}

    conns2 = [{(0, 4): odynn.circuit.SYNAPSE,
               (1, 4): odynn.circuit.SYNAPSE,
               (2, 4): odynn.circuit.SYNAPSE,
               (3, 2): odynn.circuit.SYNAPSE,
               } for _ in range(4)]
    gaps2 = [{(0,3): odynn.circuit.GAP} for _ in range(4)]

    def test_calculate(self):
        print('#1 batch')
        c = Circuit(PyBioNeuron([PyBioNeuron.default_params for _ in range(5)], 0.1), self.conns, self.gaps)
        i = np.ones((7,5))
        st, cur = c.calculate(i)
        self.assertEqual(st.shape[0], 7)
        self.assertEqual(st.shape[1], len(PyBioNeuron.default_init_state))
        self.assertEqual(st.shape[2], 5)
        print('#2 batches')
        i = np.ones((7,2,5))
        st, cur = c.calculate(i)
        self.assertEqual(st.shape[0], 7)
        self.assertEqual(st.shape[1], len(PyBioNeuron.default_init_state))
        self.assertEqual(st.shape[2], 2)
        self.assertEqual(st.shape[3], 5)

        print('#1 batch, 4 models')
        c = Circuit(PyBioNeuron([PyBioNeuron.default_params for _ in range(5)], 0.1), self.conns2, self.gaps2)
        i = np.ones((7, 5, 4))
        st, cur = c.calculate(i)
        self.assertEqual(st.shape[0], 7)
        self.assertEqual(st.shape[1], len(PyBioNeuron.default_init_state))
        self.assertEqual(st.shape[2], 5)
        self.assertEqual(st.shape[-1], 4)
        print('#2 batches, 4 models')
        i = np.ones((7, 2, 5, 4))
        st, cur = c.calculate(i)
        self.assertEqual(st.shape[0], 7)
        self.assertEqual(st.shape[1], len(PyBioNeuron.default_init_state))
        self.assertEqual(st.shape[2], 2)
        self.assertEqual(st.shape[3], 5)
        self.assertEqual(st.shape[-1], 4)




