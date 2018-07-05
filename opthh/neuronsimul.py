"""
.. module:: neuronsimul
    :synopsis: Module for simulation of neurons

.. moduleauthor:: Marc Javin
"""

from .neuron import NeuronFix, NeuronTf
import time
from . import hhmodel, datas, utils
import numpy as np
import pickle


class NeuronSimul():
    """Full Hodgkin-Huxley Model implemented in Python"""

    def __init__(self, init_p=None, t=datas.t, i_inj=datas.i_inj, neuron=None):
        self.dt = t[1]-t[0]
        if(neuron is not None):
            self.neuron = neuron
        else:
            self.neuron = NeuronFix(init_p, dt=self.dt)
        self.t = t
        self.i_inj = i_inj

    def comp_pars(self, ps, show=True, save=False):
        """Compare different parameter sets on the same experiment"""
        start = time.time()
        neurons = NeuronFix(init_p=ps, dt=self.dt)
        X = neurons.calculate(self.i_inj)
        print("Simulation time : ", time.time() - start)
        neurons.plot_output(self.t, self.i_inj, X, show=show, save=save)

    def comp_pars_targ(self, p, suffix='', save=False, show=True):
        """Compare parameter sets with a target"""
        start = time.time()
        neurons = NeuronFix(init_p=p, dt=self.dt)
        X = neurons.calculate(self.i_inj)
        print("Simulation time : ", time.time() - start)
        neurons.plot_output(self.t, self.i_inj, X[:, :, :-1], X[:, :, -1], suffix=suffix, save=save, show=show, l=2, lt=2, targstyle='-.')

    def comp_neurons(self, neurons, show=True, save=False):
        """Compare different neurons on the same experiment"""
        start = time.time()
        X = []
        for n in neurons:
            X.append(n.calculate(self.i_inj))
        X = np.stack(X, axis=2)
        print("Simulation time : ", time.time() - start)
        neurons[0].plot_output(self.t, self.i_inj, X, show=show, save=save)

    def comp_neuron_trace(self, neuron, trace, scale=False, show=True, save=False):
        """Compare a neuron with a given measured trace"""
        start = time.time()
        X = neuron.calculate(self.i_inj)
        print("Simulation time : ", time.time() - start)
        if scale:
            for i, t in enumerate(trace):
                if t is not None:
                    t *= np.max(X[:,i]) / np.max(t)
        neuron.plot_output(self.t, self.i_inj, X, trace, show=show, save=save)


    """Runs and plot the neuron"""
    def simul(self, dump=False, suffix='', show=False, save=True, ca_true=None):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        start = time.time()

        #[t,s,batch]
        X = self.neuron.calculate(self.i_inj)

        print(time.time() - start)

        # if self.neuron.loop_func == self.neuron.ica_from_v:
        #     plots_ica_from_v(self.t, self.i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save)
        # elif self.neuron.loop_func == self.neuron.ik_from_v:
        #     plots_ik_from_v(self.t, self.i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save)
        # else:
        if True:
            if self.i_inj.ndim > 1:
                for i in range(self.i_inj.shape[1]):
                    self.neuron.plot_results(self.t, self.i_inj[:,i], np.array(X[:,:,i]), suffix='target_%s%s' % (suffix,i), show=show,
                                  save=save)
            else:
                self.neuron.plot_results(self.t, self.i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save, ca_true=ca_true)

        todump = [self.t, self.i_inj, X[:, self.neuron.V_pos], X[:, self.neuron.Ca_pos]]
        if dump:
            with open(datas.DUMP_FILE, 'wb') as f:
                pickle.dump(todump, f)
            return datas.DUMP_FILE
        else:
            return todump



if __name__ == '__main__':

    t,i = datas.give_train2(0.5)
    sim = NeuronSimul(init_p=hhmodel.DEFAULT, t=t, i_inj=i)
    sim.simul(show=True)

    exit(0)



    sim = NeuronSimul(init_p=hhmodel.DEFAULT, t=datas.t, i_inj=datas.i_inj)
    sim.comp_pars_targ(hhmodel.DEFAULT, hhmodel.DEFAULT_2, show=True, save=False)


    TIME=['p__tau', 'q__tau', 'n__tau', 'e__tau', 'f__tau', 'decay_ca']
    import scipy as sp
    t = np.array(sp.arange(0.0, 30., 0.001))
    i = 2.*(t==0) + 10. *  ((t > 10) & (t < 20))
    # td = np.array(sp.arange(0.0, 1., 0.01))
    # id = 10. * ((td > 0) & (td < 100)) + 1. * ((td > 2000) & (td < 3000))

    t10 = np.array(sp.arange(0.0, 300., 0.01))
    i10 = 2.*(t10==0) + 10. * ((t10 > 100) & (t10 < 200))


    p = hhmodel.DEFAULT
    p10 = dict([(var, val*10) if var in TIME else (var, val) for var, val in p.items()])

    for ti in TIME:
        print(ti)
        print(p[ti], p10[ti])

    sim = NeuronSimul(init_p=p, t=t, i_inj=i)
    print(sim.neuron.dt)
    sim.simul( suffix='yo')

    p['rho_ca'] = p['rho_ca']*10
    sim = NeuronSimul(init_p=p, t=t, i_inj=i)
    print(sim.neuron.dt)
    sim.simul(suffix='yorho')

    sim10 = NeuronSimul(init_p=p10, t=t10, i_inj=i10)
    print(sim10.neuron.dt)
    sim10.simul( suffix='yo10')


    # simdt = HH_simul(init_p=p, t=td, i_inj=id)
    # simdt.simul(suffix='yo_dt')
