from unittest import TestCase
from Neuron import HodgkinHuxley, Neuron_tf, Neuron_fix
import params
import numpy as np
import tensorflow as tf

p = params.DEFAULT

class TestHodgkinHuxley(TestCase):

    def test_init(self):
        p = params.DEFAULT
        hh = HodgkinHuxley(init_p=[p for _ in range(10)], loop_func=HodgkinHuxley.integ_comp)
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh.init_state.shape, (7,hh.num))
        self.assertIsInstance(hh.param, dict)
        self.assertEqual(list(hh.param.values())[0].shape, (10,))
        self.assertEqual(hh.param.keys(), p.keys())

        hh = HodgkinHuxley(init_p=[params.give_rand() for _ in range(13)], loop_func=HodgkinHuxley.no_tau)
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh.init_state.shape, (7, hh.num))
        self.assertIsInstance(hh.param, dict)
        self.assertEqual(list(hh.param.values())[0].shape, (13,))
        self.assertEqual(hh.param.keys(), p.keys())

        hh = HodgkinHuxley(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh.init_state.shape, (7,))
        self.assertIsInstance(hh.param, dict)
        self.assertEqual(hh.param, p)


class TestNeuron_tf(TestCase):

    def test_init(self):
        hh = Neuron_tf(init_p=[p for _ in range(10)], loop_func=HodgkinHuxley.integ_comp)
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh.init_state.shape, (7,hh.num))
        self.assertIsInstance(hh.param, dict)
        self.assertEqual(list(hh.param.values())[0].shape, (10,))
        self.assertEqual(hh.param.keys(), p.keys())

        hh = Neuron_tf(init_p=[params.give_rand() for _ in range(13)], loop_func=HodgkinHuxley.no_tau)
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh.init_state.shape, (7, hh.num))
        self.assertIsInstance(hh.param, dict)
        self.assertEqual(list(hh.param.values())[0].shape, (13,))
        self.assertEqual(hh.param.keys(), p.keys())

        hh = Neuron_tf(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh.init_state.shape, (7,))
        self.assertIsInstance(hh.param, dict)
        self.assertEqual(hh.param, p)


    def test_parallelize(self):
        n = Neuron_tf(init_p=p)
        sh = list(n.init_state.shape)
        n.parallelize(10)
        sh.append(10)
        shp = (10,)
        self.assertEqual(n.init_state.shape, tuple(sh))
        self.assertEqual(list(n.init_p.values())[0].shape, tuple(shp))

        n = Neuron_tf(init_p=[p for _ in range(8)])
        sh = list(n.init_state.shape)
        shp = list(list(n.init_p.values())[0].shape)
        n.parallelize(11)
        sh.append(11)
        shp.append(11)
        self.assertEqual(n.init_state.shape, tuple(sh))
        self.assertEqual(list(n.init_p.values())[0].shape, tuple(shp))

    def test_build_graph(self):
        n = Neuron_tf(init_p=p)
        nn = Neuron_tf(init_p=[p for _ in range(8)])
        i,res = n.build_graph()
        self.assertEqual(i.get_shape().as_list(), [None])
        i, res = n.build_graph(batch=3)
        self.assertEqual(i.get_shape().as_list(), [None, None])
        i, res = nn.build_graph()
        self.assertEqual(i.get_shape().as_list(), [None, 8])
        i, res = nn.build_graph(batch=4)
        self.assertEqual(i.get_shape().as_list(), [None, None, 8])


    def test_calculate(self):
        n = Neuron_tf(init_p=p)
        nn = Neuron_tf(init_p=[p for _ in range(8)])
        i = np.array([2., 3., 0.])
        x = n.calculate(i)
        self.assertEqual(n.init_state.shape[0], x.shape[1])
        self.assertEqual(x.shape[0], len(i))

        i = np.array([[2., 2.], [3., 3.], [0., 0.]])
        x = n.calculate(i)
        self.assertEqual(i.shape[1], x.shape[-1])
        self.assertEqual(x.shape[0], i.shape[0])  # same time
        self.assertEqual(x.shape[1], n.init_state.shape[0])
        self.assertEqual(x.shape[2], i.shape[1])  # same nb of batch



class TestNeuron_fix(TestCase):

    def test_init(self):
        hh = Neuron_fix(init_p=[p for _ in range(10)], loop_func=HodgkinHuxley.integ_comp)
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh.init_state.shape, (7,hh.num))
        self.assertIsInstance(hh.param, dict)
        self.assertEqual(list(hh.param.values())[0].shape, (10,))
        self.assertEqual(hh.param.keys(), p.keys())

        hh = Neuron_fix(init_p=[params.give_rand() for _ in range(13)], loop_func=HodgkinHuxley.no_tau)
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh.init_state.shape, (7, hh.num))
        self.assertIsInstance(hh.param, dict)
        self.assertEqual(list(hh.param.values())[0].shape, (13,))
        self.assertEqual(hh.param.keys(), p.keys())

        hh = Neuron_fix(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh.init_state.shape, (7,))
        self.assertIsInstance(hh.param, dict)
        self.assertEqual(hh.param, p)

    def test_step(self):
        hh = Neuron_fix(p)
        hh.step(2.)
        self.assertEqual(hh.init_state.shape, hh.state.shape)

        hh = Neuron_fix(init_p=[params.give_rand() for _ in range(13)])
        hh.step(2.)
        self.assertEqual(hh.init_state.shape, hh.state.shape)

    def test_calculate(self):
        hh = Neuron_fix(p)
        i = [2., 3., 0.]
        x = hh.calculate(i)
        self.assertEqual(hh.init_state.shape, hh.state.shape)
        self.assertEqual(x.shape[0], len(i))

        i = np.array([[2., 2.], [3., 3.], [0., 0.]])
        x = hh.calculate(i)
        self.assertEqual(i.shape[1], hh.state.shape[-1])
        self.assertEqual(x.shape[0], i.shape[0]) #same time
        self.assertEqual(x.shape[1], hh.init_state.shape[0])
        self.assertEqual(x.shape[2], i.shape[1]) #same nb of batch









if __name__ == '__main__':
    TestCase.main()