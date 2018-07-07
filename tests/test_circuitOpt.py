from unittest import TestCase
from context import opthh
import opthh.circuit
import opthh.datas
from opthh import hhmodel, utils
import numpy as np
from opthh.circuitopt import CircuitOpt
from opthh.circuitsimul import CircuitSimul
from opthh.utils import set_dir


class TestCircuitOpt(TestCase):

    def test_opt_circuits(self):
        n_neuron = 2
        conns = {(0, 1): opthh.circuit.SYNAPSE_inhib,
                 (1, 0): opthh.circuit.SYNAPSE_inhib}
        conns_opt = {(0, 1): opthh.circuit.get_syn_rand(False),
                     (1, 0): opthh.circuit.get_syn_rand(False)}
        conns_opt_parallel = [conns_opt for _ in range(10)]
        dir = 'unittest/'
        dir = utils.set_dir(dir)

        t, i = opthh.datas.give_train(dt=0.5, max_t=5.)
        length = int(5./0.5)
        i_1 = np.zeros(i.shape)
        i_injs = np.stack([i, i_1], axis=2)

        p = hhmodel.give_rand()
        pars = [p for _ in range(n_neuron)]

        print('one target'.center(40, '#'))
        n_out = [1]
        c = CircuitSimul(pars, conns, t, i_injs, dt=0.5)
        train = c.simul(n_out=n_out,  show=False)
        co = CircuitOpt(pars, conns_opt, dt=0.5)
        co.opt_circuits(dir, n_out=n_out, train=train, epochs=1)
        self.assertEqual(co._loss.shape, ())
        self.assertEqual(co._V.shape, (length, i_injs.shape[1], len(n_out)))
        self.assertEqual(co._X.shape, (length, i_injs.shape[1], n_neuron))

        print('one target parallel'.center(40, '#'))
        co = CircuitOpt(pars, conns_opt_parallel, dt=0.5)
        co.opt_circuits(dir, n_out=n_out, train=train, epochs=1)
        self.assertEqual(co._loss.shape, (10,))
        self.assertEqual(co._V.shape, (length, i_injs.shape[1], len(n_out), 10))
        self.assertEqual(co._X.shape, (length, i_injs.shape[1], n_neuron, 10))

        print('several targets'.center(40, '#'))
        n_out = [0,1]
        train = c.simul(n_out=n_out,  show=False)
        co = CircuitOpt(pars, conns, dt=0.5)
        co.opt_circuits(dir, n_out=n_out, train=train, epochs=1)
        self.assertEqual(co._loss.shape, ())
        self.assertEqual(co._V.shape, (length, i_injs.shape[1], len(n_out)))
        self.assertEqual(co._X.shape, (length, i_injs.shape[1], n_neuron))

        print('several targets parallel'.center(40, '#'))
        co = CircuitOpt(pars, conns_opt_parallel, dt=0.5)
        co.opt_circuits(dir, n_out=n_out, train=train, epochs=1)
        self.assertEqual(co._loss.shape, (10,))
        self.assertEqual(co._V.shape, (length, i_injs.shape[1], len(n_out), 10))
        self.assertEqual(co._X.shape, (length, i_injs.shape[1], n_neuron, 10))

