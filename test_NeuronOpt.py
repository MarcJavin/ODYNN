from unittest import TestCase
from NeuronSimul import NeuronSimul
from NeuronOpt import NeuronOpt
import neuron_params, params
import utils
from Neuron import NeuronLSTM

class TestNeuronOpt(TestCase):

    def test_optimize(self):
        utils.set_dir('unittest')
        dt = 0.5
        t,i = params.give_train(dt=dt, max_t=5.)
        default = neuron_params.DEFAULT
        pars = neuron_params.give_rand()
        sim = NeuronSimul(init_p=default, t=t, i_inj=i)
        file = sim.simul(show=False, suffix='train', dump=True)

        n = NeuronLSTM(dt=dt)
        opt = NeuronOpt(neuron=n)
        n = opt.optimize('unittest', w=[1, 1], epochs=1, file=file)

        #one neuron
        opt = NeuronOpt(init_p=pars, dt=dt)
        self.assertEqual(opt.parallel, 1)
        n = opt.optimize('unittest', w=[1,1], epochs=1, file=file)

        #parallel
        pars = [neuron_params.give_rand() for _ in range(2)]
        opt = NeuronOpt(init_p=pars, dt=dt)
        self.assertEqual(opt.parallel, 2)
        n = opt.optimize('unittest', w=[1, 1], epochs=1, file=file)
        self.assertEqual(opt.loss.shape[0], opt.parallel)

