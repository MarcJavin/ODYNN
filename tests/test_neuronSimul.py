from unittest import TestCase
from context import opthh
from opthh.neuronsimul import NeuronSimul
from opthh import config, datas
from opthh.neuron import NeuronTf, NeuronFix


class TestNeuronSimul(TestCase):

    pars = config.NEURON_MODEL.default_params
    pars5 = [pars for _ in range(5)]
    dt = 0.5
    t, i = datas.give_train(dt=dt, max_t=5.)
    sim = NeuronSimul(pars, t=t, i_inj=i)

    def test_comp(self):
        self.sim.comp(self.pars5)

    def test_comp_targ(self):
        self.sim.comp_targ(self.pars5, self.pars)

    def test_Sim(self):
        self.sim.simul()
        ntf = NeuronTf(self.pars)
        nfix = NeuronFix(self.pars)
        simtf = NeuronSimul(ntf, t=self.t, i_inj=self.i)
        simfix = NeuronSimul(nfix, t=self.t, i_inj=self.i)
        restf = simtf.simul(dump=False)
        resfix = simfix.simul(dump=False)
        self.assertEqual(resfix.all(), restf.all())