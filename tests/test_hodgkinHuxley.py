from unittest import TestCase
from opthh.neuron import HodgkinHuxley, NeuronTf, NeuronFix
from opthh import hhmodel
import numpy as np

p = hhmodel.DEFAULT


class HodgkinHuxley2(HodgkinHuxley):

    def calculate(self, i):
        pass


class TestHodgkinHuxley(TestCase):

    def test_init(self):
        p = hhmodel.DEFAULT
        hh = HodgkinHuxley2(init_p=[p for _ in range(10)])
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh.init_state.shape, (7,hh.num))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(list(hh._param.values())[0].shape, (10,))
        self.assertEqual(hh._param.keys(), p.keys())

        hh = HodgkinHuxley2(init_p=[hhmodel.give_rand() for _ in range(13)])
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh.init_state.shape, (7, hh.num))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(list(hh._param.values())[0].shape, (13,))
        self.assertEqual(hh._param.keys(), p.keys())

        hh = HodgkinHuxley2(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh.init_state.shape, (7,))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(hh._param, p)


class TestNeuronTf(TestCase):

    def test_init(self):
        hh = NeuronTf(init_p=[p for _ in range(10)])
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh.init_state.shape, (7,hh.num))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(list(hh._param.values())[0].shape, (10,))
        self.assertEqual(hh._param.keys(), p.keys())

        hh = NeuronTf(init_p=[hhmodel.give_rand() for _ in range(13)])
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh.init_state.shape, (7, hh.num))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(list(hh._param.values())[0].shape, (13,))
        self.assertEqual(hh._param.keys(), p.keys())

        hh = NeuronTf(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh.init_state.shape, (7,))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(hh._param, p)

    def test_parallelize(self):
        n = NeuronTf(init_p=p)
        sh = list(n.init_state.shape)
        n.parallelize(10)
        sh.append(10)
        shp = (10,)
        self.assertEqual(n.init_state.shape, tuple(sh))
        self.assertEqual(list(n.init_p.values())[0].shape, tuple(shp))

        n = NeuronTf(init_p=[p for _ in range(8)])
        sh = list(n.init_state.shape)
        shp = list(list(n.init_p.values())[0].shape)
        n.parallelize(11)
        sh.append(11)
        shp.append(11)
        self.assertEqual(n.init_state.shape, tuple(sh))
        self.assertEqual(list(n.init_p.values())[0].shape, tuple(shp))

    def test_build_graph(self):
        n = NeuronTf(init_p=p)
        nn = NeuronTf(init_p=[p for _ in range(8)])
        i,res = n.build_graph()
        self.assertEqual(i.get_shape().as_list(), [None])
        i, res = n.build_graph(batch=3)
        self.assertEqual(i.get_shape().as_list(), [None, None])
        i, res = nn.build_graph()
        self.assertEqual(i.get_shape().as_list(), [None, 8])
        i, res = nn.build_graph(batch=4)
        self.assertEqual(i.get_shape().as_list(), [None, None, 8])

    def test_calculate(self):
        n = NeuronTf(init_p=p)
        nn = NeuronTf(init_p=[p for _ in range(8)])
        i = np.array([2., 3., 0.])
        ii = np.array([[2., 2.], [3., 3.], [0., 0.]])

        x = n.calculate(i)
        self.assertEqual(n.init_state.shape[0], x.shape[1])
        self.assertEqual(x.shape[0], len(i))

        x = n.calculate(ii)
        self.assertEqual(ii.shape[1], x.shape[2])
        self.assertEqual(x.shape[0], ii.shape[0])  # same time
        self.assertEqual(x.shape[1], n.init_state.shape[0])
        self.assertEqual(x.shape[2], ii.shape[1])  # same nb of batch

        x = nn.calculate(i) #several neurons, one batch
        self.assertEqual(x.shape[-1], nn.num)
        self.assertEqual(x.shape[0], len(i))
        self.assertEqual(x.shape[1], nn.init_state.shape[0])

        xx2 = nn.calculate(ii) #several neurons, several batches
        xx = nn.calculate(np.stack([ii for _ in range(8)], axis=ii.ndim))  # several neurons, several batches
        self.assertEqual(xx.shape[-1], nn.num)
        self.assertEqual(xx.shape[0], ii.shape[0])  # same time
        self.assertEqual(xx.shape[1], nn.init_state.shape[0])
        self.assertEqual(xx.shape[2], ii.shape[1])  # same nb of batch
        self.assertEqual(xx.all(), xx2.all())


class TestNeuronFix(TestCase):

    def test_init(self):
        hh = NeuronFix(init_p=[p for _ in range(10)])
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh.init_state.shape, (7,hh.num))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(list(hh._param.values())[0].shape, (10,))
        self.assertEqual(hh._param.keys(), p.keys())

        hh = NeuronFix(init_p=[hhmodel.give_rand() for _ in range(13)])
        self.assertEqual(len(hh.get_init_state()), 7)
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh.init_state.shape, (7, hh.num))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(list(hh._param.values())[0].shape, (13,))
        self.assertEqual(hh._param.keys(), p.keys())

        hh = NeuronFix(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh.init_state.shape, (7,))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(hh._param, p)

    def test_step(self):
        hh = NeuronFix(p)
        hh.step(2.)
        self.assertEqual(hh.init_state.shape, hh.state.shape)

        hh = NeuronFix(init_p=[hhmodel.give_rand() for _ in range(13)])
        hh.step(2.)
        self.assertEqual(hh.init_state.shape, hh.state.shape)

    def test_calculate(self):
        hh = NeuronFix(p)
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