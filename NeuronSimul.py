from Neuron import NeuronFix, NeuronTf, V_pos, Ca_pos
import time
from utils import plots_results, plots_ik_from_v, plots_ica_from_v
import utils
import data
import neuron_params, params
import numpy as np
import pickle


class NeuronSimul():
    """Full Hodgkin-Huxley Model implemented in Python"""

    def __init__(self, init_p=neuron_params.DEFAULT, t=params.t, i_inj=params.i_inj):
        self.dt = t[1]-t[0]
        self.neuron = NeuronFix(init_p, dt=self.dt)
        self.t = t
        self.i_inj = i_inj


    """Compare different parameter sets on the same experiment"""
    def comp(self, ps, show=True, save=False):
        start = time.time()
        neurons = NeuronTf(init_p=ps, dt=self.dt)
        X = neurons.calculate(self.i_inj)
        Vs = X[:,V_pos]
        Cacs = X[:,Ca_pos]
        print(time.time() - start)
        utils.plots_output_mult(self.t, self.i_inj, Vs, Cacs, show=show, save=save)

    """Compare 2 parameters sets"""
    def comp_targ(self, p, p_targ, suffix='', save=False, show=True):
        neurons = NeuronTf(init_p=[p, p_targ], dt=self.dt)
        X = neurons.calculate(self.i_inj)

        utils.plots_output_double(self.t, self.i_inj, X[:,V_pos, 0], X[:,V_pos, -1], X[:,Ca_pos, 0], X[:,Ca_pos, -1], suffix=suffix, save=save, show=show, l=2, lt=2, targstyle='-.')


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
                    plots_results(self.neuron, self.t, self.i_inj[:,i], np.array(X[:,:,i]), suffix='target_%s%s' % (suffix,i), show=show,
                                  save=save)
            else:
                plots_results(self.neuron, self.t, self.i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save, ca_true=ca_true)

        if dump:
            todump = [self.t, self.i_inj, X[:, V_pos], X[:, Ca_pos]]
            with open(data.DUMP_FILE, 'wb') as f:
                pickle.dump(todump, f)
            return data.DUMP_FILE



if __name__ == '__main__':



    sim = NeuronSimul(init_p=neuron_params.DEFAULT, t=neuron_params.t, i_inj=neuron_params.i_inj)
    sim.comp_targ(neuron_params.DEFAULT, neuron_params.DEFAULT_2, show=True, save=False)


    TIME=['p__tau', 'q__tau', 'n__tau', 'e__tau', 'f__tau', 'decay_ca']
    import scipy as sp
    t = np.array(sp.arange(0.0, 30., 0.001))
    i = 2.*(t==0) + 10. *  ((t > 10) & (t < 20))
    # td = np.array(sp.arange(0.0, 1., 0.01))
    # id = 10. * ((td > 0) & (td < 100)) + 1. * ((td > 2000) & (td < 3000))

    t10 = np.array(sp.arange(0.0, 300., 0.01))
    i10 = 2.*(t10==0) + 10. * ((t10 > 100) & (t10 < 200))


    p = neuron_params.DEFAULT
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
