from unittest import TestCase
from context import opthh
from opthh import config, utils, datas
from opthh.neuron import NeuronLSTM
from opthh.neuropt import NeuronOpt
from opthh.neursimul import simul


class TestNeuronOpt(TestCase):

    def test_optimize(self):
        dir = utils.set_dir('unittest')
        dt = 0.5
        t,i = datas.give_train(dt=dt, max_t=5.)
        default = config.NEURON_MODEL.default_params
        pars = config.NEURON_MODEL.get_random()
        train = simul(p=default, dt=dt, i_inj=i, show=False, suffix='train')
        plot=False

        print('LSTM'.center(40, '#'))
        n = NeuronLSTM(dt=dt)
        opt = NeuronOpt(neuron=n)
        self.assertEqual(opt._parallel, 1)
        n = opt.optimize(dir, w=[1, 1],  train=train, epochs=1, plot=plot)

        print('One neuron'.center(40, '#'))
        opt = NeuronOpt(init_p=pars, dt=dt)
        self.assertEqual(opt._parallel, 1)
        n = opt.optimize(dir, w=[1,1],  train=train, epochs=1, plot=plot)
        print('One neuron reload'.center(40, '#'))
        n = opt.optimize(dir, w=[1, 1],  reload=True, train=train, epochs=1, plot=plot)
        print('One neuron with test'.center(40, '#'))
        n = opt.optimize(dir, w=[1, 1], train=train, test=train, epochs=1, plot=plot)


        print('Parallel'.center(40, '#'))
        pars = [config.NEURON_MODEL.get_random() for _ in range(2)]
        opt = NeuronOpt(init_p=pars, dt=dt)
        self.assertEqual(opt._parallel, 2)
        n = opt.optimize(dir, w=[1, 1],  train=train, epochs=1, plot=plot)
        self.assertEqual(opt._loss.shape[0], opt._parallel)

