from unittest import TestCase
import odin.nsimul as sim
from odin import datas
from odin.models import cfg_model
from odin.neuron import BioNeuronFix
import numpy as np


class TestNeuronSimul(TestCase):

    pars = cfg_model.NEURON_MODEL.default_params
    pars5 = [cfg_model.NEURON_MODEL.default_params for _ in range(5)]
    dt = 0.5
    t, i = datas.give_train(dt=dt, max_t=5.)
    i = i[:,4]


    def test_comp(self):
        sim.comp_pars(self.pars5, dt=self.dt, i_inj=self.i, show=False)
        sim.comp_pars(self.pars5, t=self.t, i_inj=self.i, show=False)

    def test_comp_targ(self):
        sim.comp_pars_targ(self.pars5, self.pars, dt=self.dt, i_inj=self.i, show=False)
        sim.comp_pars_targ(self.pars, self.pars, t=self.t, i_inj=self.i, show=False)

    def test_comp_neurons(self):
        sim.comp_neurons([BioNeuronFix(self.pars) for _ in range(3)], i_inj=self.i, show=False)

    def test_comp_neur_trqce(self):
        sim.comp_neuron_trace(BioNeuronFix(self.pars), i_inj=self.i, trace=np.zeros((len(cfg_model.NEURON_MODEL.default_init_state),len(self.i))), show=False)
        sim.comp_neuron_trace(BioNeuronFix(self.pars), i_inj=self.i, scale=True,
                              trace=np.zeros((len(cfg_model.NEURON_MODEL.default_init_state), len(self.i))), show=False)

    def test_Sim(self):
        nfix = BioNeuronFix(self.pars)
        sim.simul(neuron=nfix, dt=self.dt, i_inj=self.i)