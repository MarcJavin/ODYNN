from unittest import TestCase
from context import opthh
from opthh import config, utils, datas
from opthh.neuron import NeuronLSTM
from opthh.neuronopt import NeuronOpt
from opthh.neuronsimul import NeuronSimul


class TestNeuronOpt(TestCase):

    def test_optimize(self):
        utils.set_dir('unittest')
        dt = 0.5
        t,i = datas.give_train(dt=dt, max_t=5.)
        default = config.NEURON_MODEL.default_params
        pars = config.NEURON_MODEL.get_random()
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
        n = opt.optimize('unittest', w=[1, 1], train=train, test=train)


        #parallel
        pars = [config.NEURON_MODEL.get_random() for _ in range(2)]
        opt = NeuronOpt(init_p=pars, dt=dt, epochs=1)
        self.assertEqual(opt.parallel, 2)
        n = opt.optimize('unittest', w=[1, 1],  train=train)
        self.assertEqual(opt.loss.shape[0], opt.parallel)

