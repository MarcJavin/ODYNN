from unittest import TestCase
from context import opthh
import opthh.neursimul as sim
from opthh import config, datas
from opthh.neuron import BioNeuronTf, BioNeuronFix


class TestNeuronSimul(TestCase):

    pars = config.NEURON_MODEL.default_params
    pars5 = [config.NEURON_MODEL.default_params for _ in range(5)]
    dt = 0.5
    t, i = datas.give_train(dt=dt, max_t=5.)
    i = i[:,4]


    def test_comp(self):
        sim.comp_pars(self.pars5, dt=self.dt, i_inj=self.i, show=False)

    def test_comp_targ(self):
        sim.comp_pars_targ(self.pars5, self.pars, dt=self.dt, i_inj=self.i, show=False)

    def test_Sim(self):
        sim.simul()
        nfix = BioNeuronFix(self.pars)
        resfix = sim.simul(neuron=nfix, dt=self.dt, i_inj=self.i, dump=False)