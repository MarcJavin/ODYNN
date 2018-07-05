from unittest import TestCase
from context import opthh
from opthh.neuronsimul import NeuronSimul
from opthh import config, datas
from opthh.neuron import NeuronTf, NeuronFix


class TestNeuronSimul(TestCase):

    pars = config.NEURON_MODEL.default_params
    pars5 = [config.NEURON_MODEL.default_params for _ in range(5)]
    dt = 0.5
    t, i = datas.give_train(dt=dt, max_t=5.)
    sim = NeuronSimul(pars, t=t, i_inj=i)

    def test_comp(self):
        self.sim.comp_pars(self.pars5, show=False)

    def test_comp_targ(self):
        self.sim.comp_pars_targ(self.pars5, show=False)

    def test_Sim(self):
        self.sim.simul()
        nfix = NeuronFix(self.pars)
        simfix = NeuronSimul(neuron=nfix, t=self.t, i_inj=self.i)
        resfix = simfix.simul(dump=False)