from unittest import TestCase

from opthh import hhmodel, params, utils
from opthh.neuron import NeuronLSTM
from opthh.neuronopt import NeuronOpt
from opthh.neuronsimul import NeuronSimul


class TestNeuronOpt(TestCase):

    def test_optimize(self):
        utils.set_dir('unittest')
        dt = 0.5
        t,i = params.give_train(dt=dt, max_t=5.)
        default = hhmodel.DEFAULT
        pars = hhmodel.give_rand()
        sim = NeuronSimul(init_p=default, t=t, i_inj=i)
        file = sim.simul(show=False, suffix='train', dump=True)

        #LSTM
        n = NeuronLSTM(dt=dt)
        opt = NeuronOpt(neuron=n)
        self.assertEqual(opt.parallel, 1)
        n = opt.optimize('unittest', w=[1, 1], epochs=1, file=file)

        #one neuron
        opt = NeuronOpt(init_p=pars, dt=dt)
        self.assertEqual(opt.parallel, 1)
        n = opt.optimize('unittest', w=[1,1], epochs=1, file=file)
        n = opt.optimize('unittest', w=[1, 1], epochs=1, reload=True, file=file)

        #parallel
        pars = [hhmodel.give_rand() for _ in range(2)]
        opt = NeuronOpt(init_p=pars, dt=dt)
        self.assertEqual(opt.parallel, 2)
        n = opt.optimize('unittest', w=[1, 1], epochs=1, file=file)
        self.assertEqual(opt.loss.shape[0], opt.parallel)

