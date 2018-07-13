from unittest import TestCase
from context import opthh
import opthh.circuit as cr
import opthh.datas
from opthh import hhmodel, utils, config
import numpy as np
from opthh.circopt import CircuitOpt
import opthh.circsimul as csim
import opthh.neuron as nr


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

        dt = 0.5
        t, i = opthh.datas.give_train(dt=dt, max_t=5.)
        length = int(5./0.5)
        i_1 = np.zeros(i.shape)
        i_injs = np.stack([i, i_1], axis=2)

        p = hhmodel.give_rand()
        pars = [p for _ in range(n_neuron)]
        plot = False

        print('one target'.center(40, '#'))
        n_out = [1]
        train = csim.simul(pars, conns, t, i_injs, n_out=n_out,  show=False)
        co = CircuitOpt(pars, conns_opt, dt=dt)
        co.opt_circuits(dir, n_out=n_out, train=train, epochs=1, plot=plot)
        self.assertEqual(co._loss.shape, ())
        self.assertEqual(co._V.shape, (length, i_injs.shape[1], len(n_out)))
        self.assertEqual(co._X.shape, (length, i_injs.shape[1], n_neuron))

        print('one target parallel'.center(40, '#'))
        co = CircuitOpt(pars, conns_opt_parallel, dt=dt)
        co.opt_circuits(dir, n_out=n_out, train=train, epochs=1, plot=plot)
        self.assertEqual(co._loss.shape, (10,))
        self.assertEqual(co._V.shape, (length, i_injs.shape[1], len(n_out), 10))
        self.assertEqual(co._X.shape, (length, i_injs.shape[1], n_neuron, 10))

        print('several targets'.center(40, '#'))
        n_out = [0,1]
        train = csim.simul(pars, conns, t, i_injs, n_out=n_out,  show=False)
        co = CircuitOpt(pars, conns, dt=dt)
        co.opt_circuits(dir, n_out=n_out, train=train, epochs=1, plot=plot)
        self.assertEqual(co._loss.shape, ())
        self.assertEqual(co._V.shape, (length, i_injs.shape[1], len(n_out)))
        self.assertEqual(co._X.shape, (length, i_injs.shape[1], n_neuron))

        print('several targets parallel'.center(40, '#'))
        co = CircuitOpt(pars, conns_opt_parallel, dt=dt)
        co.opt_circuits(dir, n_out=n_out, train=train, epochs=1, plot=plot)
        self.assertEqual(co._loss.shape, (10,))
        self.assertEqual(co._V.shape, (length, i_injs.shape[1], len(n_out), 10))
        self.assertEqual(co._X.shape, (length, i_injs.shape[1], n_neuron, 10))

        print('1 LSTM'.center(40, '#'))
        neurons = nr.Neurons(
            [nr.NeuronLSTM(dt=dt), nr.BioNeuronTf(p, fixed='all', dt=dt)])
        c = cr.CircuitTf(neurons=neurons, conns=conns_opt)
        co = CircuitOpt(circuit=c)
        co.opt_circuits(dir, train=train, n_out=[0, 1], l_rate=(0.01, 9, 0.95), epochs=1, plot=plot)

        print('2 LSTM'.center(40, '#'))
        neurons = nr.Neurons(
            [nr.NeuronLSTM(dt=dt), nr.NeuronLSTM(dt=dt)])
        c = cr.CircuitTf(neurons=neurons, conns=conns_opt)
        co = CircuitOpt(circuit=c)
        co.opt_circuits(dir, train=train, n_out=[0, 1], l_rate=(0.01, 9, 0.95), epochs=1, plot=plot)

