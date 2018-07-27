from unittest import TestCase
from odin.neuron import BioNeuronTf, BioNeuronFix
from odin import utils
from odin.models.celeg import CElegansNeuron
from odin.models import cfg_model, celeg
import numpy as np
import pickle

p = cfg_model.NEURON_MODEL.default_params


class CElegansNeuron2(CElegansNeuron):

    def calculate(self, i):
        pass


class TestHodgkinHuxley(TestCase):

    def test_init(self):
        p = celeg.DEFAULT
        hh = CElegansNeuron2(init_p=[p for _ in range(10)])
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),hh.num))
        self.assertIsInstance(hh._init_p, dict)
        self.assertEqual(list(hh._init_p.values())[0].shape, (10,))
        self.assertEqual(hh._init_p.keys(), p.keys())

        hh = CElegansNeuron2(init_p=[celeg.CElegansNeuron.get_random() for _ in range(13)])
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh._init_p, dict)
        self.assertEqual(list(hh._init_p.values())[0].shape, (13,))
        self.assertEqual(hh._init_p.keys(), p.keys())

        hh = CElegansNeuron2(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
        self.assertIsInstance(hh._init_p, dict)
        self.assertEqual(hh._init_p, p)


class TestNeuronTf(TestCase):

    dir = utils.set_dir('unittest')

    def test_init(self):
        hh = BioNeuronTf(init_p=[p for _ in range(10)])
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),hh.num))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(list(hh.init_params.values())[0].shape, (10,))
        self.assertEqual(hh.init_params.keys(), p.keys())

        hh = BioNeuronTf(init_p={var: [val for _ in range(10)] for var, val in p.items()})
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(list(hh.init_params.values())[0].shape, (10,))
        self.assertEqual(hh.init_params.keys(), p.keys())

        hh = BioNeuronTf(init_p=[cfg_model.NEURON_MODEL.get_random() for _ in range(13)])
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(list(hh.init_params.values())[0].shape, (13,))
        self.assertEqual(hh.init_params.keys(), p.keys())

        hh = BioNeuronTf(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(hh.init_params, p)

        hh = BioNeuronTf(n_rand=15)
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 15)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(list(hh.init_params.values())[0].shape, (15,))

    def test_pickle(self):
        hh = BioNeuronTf(init_p=[p for _ in range(10)])
        with open(self.dir + 'yeee', 'wb') as f:
            pickle.dump(hh, f)
        with open(self.dir + 'yeee', 'rb') as f:
            hh = pickle.load(f)
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(list(hh.init_params.values())[0].shape, (10,))
        self.assertEqual(hh.init_params.keys(), p.keys())

        hh = BioNeuronTf(init_p={var: [val for _ in range(10)] for var, val in p.items()})
        with open(self.dir + 'yeee', 'wb') as f:
            pickle.dump(hh, f)
        with open(self.dir + 'yeee', 'rb') as f:
            hh = pickle.load(f)
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(list(hh.init_params.values())[0].shape, (10,))
        self.assertEqual(hh.init_params.keys(), p.keys())

        hh = BioNeuronTf(init_p=[cfg_model.NEURON_MODEL.get_random() for _ in range(13)])
        with open(self.dir + 'yeee', 'wb') as f:
            pickle.dump(hh, f)
        with open(self.dir + 'yeee', 'rb') as f:
            hh = pickle.load(f)
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(list(hh.init_params.values())[0].shape, (13,))
        self.assertEqual(hh.init_params.keys(), p.keys())

        hh = BioNeuronTf(n_rand=15)
        with open(self.dir + 'yeee', 'wb') as f:
            pickle.dump(hh, f)
        with open(self.dir + 'yeee', 'rb') as f:
            hh = pickle.load(f)
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 15)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(list(hh.init_params.values())[0].shape, (15,))

        hh = BioNeuronTf(p)
        with open(self.dir + 'yeee', 'wb') as f:
            pickle.dump(hh, f)
        with open(self.dir + 'yeee', 'rb') as f:
            hh = pickle.load(f)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
        self.assertIsInstance(hh.init_params, dict)
        self.assertEqual(hh.init_params, p)


    def test_parallelize(self):
        n = BioNeuronTf(init_p=p)
        sh = list(n._init_state.shape)
        n.parallelize(10)
        sh.append(10)
        shp = (10,)
        self.assertEqual(n._init_state.shape, tuple(sh))
        self.assertEqual(list(n.init_params.values())[0].shape, tuple(shp))

        n = BioNeuronTf(init_p=[p for _ in range(8)])
        sh = list(n._init_state.shape)
        shp = list(list(n.init_params.values())[0].shape)
        n.parallelize(11)
        sh.append(11)
        shp.append(11)
        self.assertEqual(n._init_state.shape, tuple(sh))
        self.assertEqual(list(n.init_params.values())[0].shape, tuple(shp))

    def test_build_graph(self):
        n = BioNeuronTf(init_p=p)
        nn = BioNeuronTf(init_p=[p for _ in range(8)])
        i,res = n.build_graph()
        self.assertEqual(i.get_shape().as_list(), [None])
        i, res = n.build_graph(batch=3)
        self.assertEqual(i.get_shape().as_list(), [None, None])
        i, res = nn.build_graph()
        self.assertEqual(i.get_shape().as_list(), [None, 8])
        i, res = nn.build_graph(batch=4)
        self.assertEqual(i.get_shape().as_list(), [None, None, 8])

    def test_calculate(self):
        n = BioNeuronTf(init_p=p)
        nn = BioNeuronTf(init_p=[p for _ in range(8)])
        i = np.array([2., 3., 0.])
        ii = np.array([[2., 2.], [3., 3.], [0., 0.]])

        x = n.calculate(i)
        self.assertEqual(n._init_state.shape[0], x.shape[1])
        self.assertEqual(x.shape[0], len(i))

        x = n.calculate(ii)
        self.assertEqual(ii.shape[1], x.shape[2])
        self.assertEqual(x.shape[0], ii.shape[0])  # same time
        self.assertEqual(x.shape[1], n._init_state.shape[0])
        self.assertEqual(x.shape[2], ii.shape[1])  # same nb of batch

        x = nn.calculate(i) #several neurons, one batch
        self.assertEqual(x.shape[-1], nn.num)
        self.assertEqual(x.shape[0], len(i))
        self.assertEqual(x.shape[1], nn._init_state.shape[0])

        xx2 = nn.calculate(ii) #several neurons, several batches
        xx = nn.calculate(np.stack([ii for _ in range(8)], axis=ii.ndim))  # several neurons, several batches
        self.assertEqual(xx.shape[-1], nn.num)
        self.assertEqual(xx.shape[0], ii.shape[0])  # same time
        self.assertEqual(xx.shape[1], nn._init_state.shape[0])
        self.assertEqual(xx.shape[2], ii.shape[1])  # same nb of batch
        self.assertEqual(xx.all(), xx2.all())


class TestNeuronFix(TestCase):

    def test_init(self):
        hh = BioNeuronFix(init_p=[p for _ in range(10)])
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),hh.num))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(list(hh._param.values())[0].shape, (10,))
        self.assertEqual(hh._param.keys(), p.keys())

        hh = BioNeuronFix(init_p={var: np.array([val for _ in range(10)]) for var, val in p.items()})
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 10)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(list(hh._param.values())[0].shape, (10,))
        self.assertEqual(hh._param.keys(), p.keys())

        hh = BioNeuronFix(init_p=[cfg_model.NEURON_MODEL.get_random() for _ in range(13)])
        self.assertEqual(len(hh._init_state), len(hh.default_init_state))
        self.assertEqual(hh.num, 13)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state), hh.num))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(list(hh._param.values())[0].shape, (13,))
        self.assertEqual(hh._param.keys(), p.keys())

        hh = BioNeuronFix(p)
        self.assertEqual(hh.num, 1)
        self.assertEqual(hh._init_state.shape, (len(hh.default_init_state),))
        self.assertIsInstance(hh._param, dict)
        self.assertEqual(hh._param, p)

    def test_step(self):
        hh = BioNeuronFix(p)
        x = hh.step(hh.init_state, 2.)
        self.assertEqual(hh._init_state.shape, x.shape)

        hh = BioNeuronFix(init_p=[cfg_model.NEURON_MODEL.get_random() for _ in range(13)])
        x = hh.step(hh.init_state, 2.)
        self.assertEqual(hh._init_state.shape, x.shape)

    def test_calculate(self):
        hh = BioNeuronFix(p)
        i = [2., 3., 0.]
        x = hh.calculate(i)
        self.assertEqual(hh._init_state.shape, x[-1].shape)
        self.assertEqual(x.shape[0], len(i))

        i = np.array([[2., 2.], [3., 3.], [0., 0.]])
        x = hh.calculate(i)
        self.assertEqual(i.shape[1], x[-1].shape[1])
        self.assertEqual(x.shape[0], i.shape[0]) #same time
        self.assertEqual(x.shape[1], hh._init_state.shape[0])
        self.assertEqual(x.shape[2], i.shape[1]) #same nb of batch




if __name__ == '__main__':
    TestCase.main()