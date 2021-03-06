from unittest import TestCase
from odynn import utils, datas
from odynn.neuron import NeuronLSTM, BioNeuronTf, PyBioNeuron
from odynn.noptim import NeuronOpt
from odynn.nsimul import simul
from odynn import optim
import tensorflow as tf
import numpy as np

dir = utils.set_dir('unittest')
dt = 0.5
t,i = datas.give_train(dt=dt, max_t=5.)
default = PyBioNeuron.default_params
pars = PyBioNeuron.get_random()
train = simul(p=default, dt=dt, i_inj=i, show=False, suffix='train')
plot=False
nr = BioNeuronTf(init_p=pars, dt=dt)

class TestNeuronOpt(TestCase):

    def test_init(self):
        with self.assertRaises(TypeError):
            no = NeuronOpt(5)

    def test_loss(self):
        co = NeuronOpt(nr)
        res = tf.zeros((len(t), len(PyBioNeuron.default_init_state), 3))
        ys_ = [tf.placeholder(shape=(len(t), 3), dtype=tf.float32, name="test")] + [
            tf.placeholder(shape=(len(t), 3), dtype=tf.float32, name="test") for _ in
            PyBioNeuron.ions.items()]
        w = [1] + [1 for _ in PyBioNeuron.ions.items()]
        co._build_loss(res, ys_, w)

    def test_settings(self):
        co = NeuronOpt(nr)
        train = [np.zeros(2), np.zeros(2), [None, None, None]]
        co.l_rate = (0.1,0.1,0.1)
        co.n_batch = 3
        w = (1.,0,0)
        co.settings(w, train)

    def test_optimize(self):
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
        optim.get_model(dir)
        optim.get_vars_all(dir)
        t, tt = optim.get_data(dir)
        self.assertEqual(t[0].all(), train[0].all())
        self.assertEqual(t[1].all(), train[1].all())
        for i, tt in enumerate(t[-1]):
            if tt is not None:
                self.assertEqual(tt.all(), train[-1][i].all())


        print('One neuron'.center(40, '#'))
        opt = NeuronOpt(nr)
        self.assertEqual(opt._parallel, 1)
        n = opt.optimize(dir, w=w,  train=train, epochs=1, plot=plot)
        print('One neuron reload'.center(40, '#'))
        n = opt.optimize(dir, w=w,  reload=True, train=train, epochs=1, plot=plot)
        print('One neuron with test'.center(40, '#'))
        n = opt.optimize(dir, w=w, train=train, test=train, epochs=1, plot=plot)


        print('Parallel'.center(40, '#'))
        pars = [PyBioNeuron.get_random() for _ in range(2)]
        opt = NeuronOpt(BioNeuronTf(init_p=pars, dt=dt))
        self.assertEqual(opt._parallel, 2)
        n = opt.optimize(dir, w=w,  train=train, epochs=1, plot=plot)
        self.assertEqual(opt._loss.shape[0], opt._parallel)

