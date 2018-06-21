from unittest import TestCase
from Neuron import HodgkinHuxley
from Neuron_simul import HH_simul
from Neuron_opt import HH_opt
import params


class TestHH_opt(TestCase):

    def test_optimize(self):
        dt = 0.5
        t,i = params.give_train(dt=dt, max_t=50.)
        default = params.DEFAULT
        pars = params.give_rand()

        loop_func = HodgkinHuxley.integ_comp
        opt = HH_opt(init_p=pars, dt=dt)
        self.assertEqual(opt.parallel, 1)
        sim = HH_simul(init_p=default, t=t, i_inj=i, loop_func=loop_func)
        file = sim.simul(show=False, suffix='train', dump=True)
        n = opt.optimize('unittest', w=[1,1], epochs=2, file=file)

        pars = [params.give_rand() for _ in range(10)]
        opt = HH_opt(init_p=pars, dt=dt)
        self.assertEqual(opt.parallel, 10)
        n = opt.optimize('unittest', w=[1, 1], epochs=2, file=file)
        self.assertEqual(opt.loss.shape[0], opt.parallel)
