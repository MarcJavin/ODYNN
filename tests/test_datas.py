from unittest import TestCase
from odin import datas


class TestDatas(TestCase):

    def test_check_alpha(self):
        datas.check_alpha(show=False)

    def test_get_real_data(self):
        dt=0.3
        ft = 2000.
        train, test = datas.get_real_data(delta=500, final_time=ft, dt=dt)
        t, i, [v, ca] = train
        self.assertEqual(len(t), round(ft/dt))
        self.assertEqual(len(t), len(i))
        self.assertEqual(v, None)
        self.assertEqual(len(ca), len(t))
        t, i, [v, ca] = test
        self.assertEqual(len(t), len(i))
        self.assertEqual(v, None)
        self.assertEqual(len(ca), len(t))

        dt = 0.5
        ft = 3500.
        train, test = datas.get_real_data(delta=500, final_time=ft, dt=dt)
        t, i, [v, ca] = train
        self.assertEqual(len(t), round(ft / dt))
        self.assertEqual(len(t), len(i))
        self.assertEqual(v, None)
        self.assertEqual(len(ca), len(t))
        t, i, [v, ca] = test
        self.assertEqual(len(t), len(i))
        self.assertEqual(v, None)
        self.assertEqual(len(ca), len(t))


    def test_give_train(self):
        t,i = datas.give_train(dt=0.2, nb_neuron_zero=None, max_t=1200.)
        self.assertEqual(t[-1], 1200.-0.2)
        self.assertEqual(t[1] - t[0], 0.2)
        self.assertEqual(i.shape[0], 1200./0.2)
        self.assertEqual(i.ndim, 2)

        t, i = datas.give_train(dt=0.5, nb_neuron_zero=3, max_t=800.)
        self.assertEqual(t[-1], 800.-0.5)
        self.assertEqual(t[1] - t[0], 0.5)
        self.assertEqual(i.shape[0], 800. / 0.5)
        self.assertEqual(i.shape[2], 4)
        self.assertEqual(i.ndim, 3)

    def test_give_test(self):
        t,i = datas.give_test(dt=0.2, max_t=1200.)
        self.assertEqual(t[-1], 1200.-0.2)
        self.assertEqual(t[1] - t[0], 0.2)
        self.assertEqual(i.shape[0], 1200./0.2)
        self.assertEqual(i.ndim, 2)

        t, i = datas.give_test(dt=0.5, max_t=800.)
        self.assertEqual(t[-1], 800.-0.5)
        self.assertEqual(t[1] - t[0], 0.5)
        self.assertEqual(i.shape[0], 800. / 0.5)

    def test_full4(self):
        t, i = datas.full4(dt=0.2, nb_neuron_zero=None, max_t=1200.)
        self.assertEqual(t[-1], 1200.-0.2)
        self.assertEqual(t[1] - t[0], 0.2)
        self.assertEqual(i.shape[0], 1200. / 0.2)
        self.assertEqual(i.shape[2], 4)
        self.assertEqual(i.ndim, 3)

        t, i = datas.full4(dt=0.5, nb_neuron_zero=3, max_t=800.)
        self.assertEqual(t[-1], 800.-0.5)
        self.assertEqual(t[1] - t[0], 0.5)
        self.assertEqual(i.shape[0], 800. / 0.5)
        self.assertEqual(i.shape[2], 7)
        self.assertEqual(i.ndim, 3)

    def test_full4_test(self):
        t, i = datas.full4_test(dt=0.2, nb_neuron_zero=None, max_t=1200.)
        self.assertEqual(t[-1], 1200.-0.2)
        self.assertEqual(t[1] - t[0], 0.2)
        self.assertEqual(i.shape[0], 1200. / 0.2)
        self.assertEqual(i.shape[2], 4)
        self.assertEqual(i.ndim, 3)

        t, i = datas.full4_test(dt=0.5, nb_neuron_zero=3, max_t=800.)
        self.assertEqual(t[-1], 800.-0.5)
        self.assertEqual(t[1] - t[0], 0.5)
        self.assertEqual(i.shape[0], 800. / 0.5)
        self.assertEqual(i.shape[2], 7)
        self.assertEqual(i.ndim, 3)