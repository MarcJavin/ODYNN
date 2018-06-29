from unittest import TestCase
from context import opthh
from opthh import hhmodel, utils, datas
from opthh.neuron import NeuronLSTM
from opthh.neuronopt import NeuronOpt
from opthh.neuronsimul import NeuronSimul


class TestNeuronOpt(TestCase):

    def test_optimize(self):
        utils.set_dir('unittest')
        dt = 0.5
        t,i = datas.give_train(dt=dt, max_t=5.)
        default = hhmodel.DEFAULT
        pars = hhmodel.give_rand()
        sim = NeuronSimul(init_p=default, t=t, i_inj=i)
        train = sim.simul(show=False, suffix='train')

        #LSTM
        n = NeuronLSTM(dt=dt)
        opt = NeuronOpt(neuron=n, epochs=1)
        self.assertEqual(opt.parallel, 1)
        n = opt.optimize('unittest', w=[1, 1],  train=train)

        #one neuron
        opt = NeuronOpt(init_p=pars, dt=dt, epochs=1)
        self.assertEqual(opt.parallel, 1)
        n = opt.optimize('unittest', w=[1,1],  train=train)
        n = opt.optimize('unittest', w=[1, 1],  reload=True, train=train)

        #parallel
        pars = [hhmodel.give_rand() for _ in range(2)]
        opt = NeuronOpt(init_p=pars, dt=dt, epochs=1)
        self.assertEqual(opt.parallel, 2)
        n = opt.optimize('unittest', w=[1, 1],  train=train)
        self.assertEqual(opt.loss.shape[0], opt.parallel)

