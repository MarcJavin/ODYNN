from unittest import TestCase
import params
import numpy as np
from Circuit_opt import Circuit_opt
from Circuit_simul import Circuit_simul
import utils

class TestCircuit_opt(TestCase):

    def test_opt_circuits(self):
        n_neuron = 2
        conns = {(0, 1): params.SYNAPSE_inhib,
                 (1, 0): params.SYNAPSE_inhib}
        conns_opt = {(0, 1): params.get_syn_rand(False),
                     (1, 0): params.get_syn_rand(False)}
        conns_opt_parallel = [conns_opt for _ in range(10)]
        dir = 'unittest'
        utils.set_dir(dir)

        t, i = params.give_train(dt=0.5, max_t=5.)
        length = int(5./0.5)
        i_1 = np.zeros(i.shape)
        i_injs = np.stack([i, i_1], axis=2)

        p = params.give_rand()
        pars = [p for _ in range(n_neuron)]

        print('one target')
        n_out = [1]
        c = Circuit_simul(pars, conns, t, i_injs, dt=0.5)
        file = c.simul(n_out=n_out, dump=True, show=False)
        co = Circuit_opt(pars, conns_opt, dt=0.5)
        co.opt_circuits(dir, n_out=n_out, file=file, epochs=1)
        self.assertEqual(co.loss.shape, ())
        self.assertEqual(co.V.shape, (length, i_injs.shape[1], len(n_out)))
        self.assertEqual(co.X.shape, (length, i_injs.shape[1], n_neuron))

        print('one target parallel')
        co = Circuit_opt(pars, conns_opt_parallel, dt=0.5)
        co.opt_circuits(dir, n_out=n_out, file=file, epochs=1)
        self.assertEqual(co.loss.shape, (10,))
        self.assertEqual(co.V.shape, (length, i_injs.shape[1], len(n_out), 10))
        self.assertEqual(co.X.shape, (length, i_injs.shape[1], n_neuron, 10))

        print('several targets')
        n_out = [0,1]
        file = c.simul(n_out=n_out, dump=True, show=False)
        co = Circuit_opt(pars, conns, dt=0.5)
        co.opt_circuits(dir, n_out=n_out, file=file, epochs=1)
        self.assertEqual(co.loss.shape, ())
        self.assertEqual(co.V.shape, (length, i_injs.shape[1], len(n_out)))
        self.assertEqual(co.X.shape, (length, i_injs.shape[1], n_neuron))

        print('several targets parallel')
        co = Circuit_opt(pars, conns_opt_parallel, dt=0.5)
        co.opt_circuits(dir, n_out=n_out, file=file, epochs=1)
        self.assertEqual(co.loss.shape, (10,))
        self.assertEqual(co.V.shape, (length, i_injs.shape[1], len(n_out), 10))
        self.assertEqual(co.X.shape, (length, i_injs.shape[1], n_neuron, 10))

