from unittest import TestCase
from Neuron import HodgkinHuxley
import params


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






if __name__ == '__main__':
    TestCase.main()