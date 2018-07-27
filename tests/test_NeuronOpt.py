from unittest import TestCase
from odin import utils, datas
from odin.models import cfg_model
from odin.neuron import NeuronLSTM, BioNeuronTf
from odin.noptim import NeuronOpt
from odin.nsimul import simul


class TestNeuronOpt(TestCase):

    def test_optimize(self):
        return
        dir = utils.set_dir('unittest')
        dt = 0.5
        t,i = datas.give_train(dt=dt, max_t=5.)
        default = cfg_model.NEURON_MODEL.default_params
        pars = cfg_model.NEURON_MODEL.get_random()
        train = simul(p=default, dt=dt, i_inj=i, show=False, suffix='train')
        plot=False

        print('LSTM'.center(40, '#'))
        n = NeuronLSTM(dt=dt)
        opt = NeuronOpt(neuron=n)
        self.assertEqual(opt._parallel, 1)
        w = [1 for _ in range(len(train[-1]))]
        n = opt.optimize(dir, w=w,  train=train, epochs=1, plot=plot)
        print('LSTM, calcium None'.center(40, '#'))
        train2 = train
        train2[-1][-1] = None
        n = opt.optimize(dir, w=w, train=train, epochs=1, plot=plot)


        print('One neuron'.center(40, '#'))
        opt = NeuronOpt(BioNeuronTf(init_p=pars, dt=dt))
        self.assertEqual(opt._parallel, 1)
        n = opt.optimize(dir, w=w,  train=train, epochs=1, plot=plot)
        print('One neuron reload'.center(40, '#'))
        n = opt.optimize(dir, w=w,  reload=True, train=train, epochs=1, plot=plot)
        print('One neuron with test'.center(40, '#'))
        n = opt.optimize(dir, w=w, train=train, test=train, epochs=1, plot=plot)


        print('Parallel'.center(40, '#'))
        pars = [cfg_model.NEURON_MODEL.get_random() for _ in range(2)]
        opt = NeuronOpt(BioNeuronTf(init_p=pars, dt=dt))
        self.assertEqual(opt._parallel, 2)
        n = opt.optimize(dir, w=w,  train=train, epochs=1, plot=plot)
        self.assertEqual(opt._loss.shape[0], opt._parallel)

