from Neuron import Neuron_fix, V_pos, Ca_pos
import time
from utils import plots_results, plots_ik_from_v, plots_ica_from_v
import utils
import data
import params
import numpy as np
import pickle


class HH_simul():
    """Full Hodgkin-Huxley Model implemented in Python"""

    def __init__(self, init_p=params.DEFAULT, t=params.t, i_inj=params.i_inj, loop_func=None):
        self.neuron = Neuron_fix(init_p, loop_func=loop_func, dt=t[1]-t[0])
        self.t = t
        self.i_inj = i_inj

    """Simulate the neuron"""
    def calculate(self):
        X = []
        self.neuron.reset()
        for i in self.i_inj:
            X.append(self.neuron.step(i))
        return np.array(X)

    """Compare different parameter sets on the same experiment"""
    def comp(self, ps):
        Vs = []
        Cacs = []
        for p in ps:
            self.neuron.param = p
            X = self.calculate()
            Vs.append(X[:,V_pos])
            Cacs.append(X[:,Ca_pos])
        utils.plots_output_mult(self.t, self.i_inj, Vs, Cacs, save=False, show=True)

    """Compare 2 parameters sets"""
    def comp_targ(self, p, p_targ):
        self.neuron.param = p
        S = self.calculate()

        self.neuron.param = p_targ
        S_targ = self.calculate()

        utils.plots_output_double(self.t, self.i_inj, S[:,V_pos], S_targ[:,V_pos], S[:,Ca_pos], S_targ[:,Ca_pos], save=False, show=True, l=2, lt=2)


    """Runs and plot the neuron"""
    def simul(self, dump=False, suffix='', show=False, save=True):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        start = time.time()

        #[t,s,batch]
        X = self.calculate()

        print(time.time() - start)

        if (self.neuron.loop_func == self.neuron.ica_from_v):
            plots_ica_from_v(self.t, self.i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save)
        elif (self.neuron.loop_func == self.neuron.ik_from_v):
            plots_ik_from_v(self.t, self.i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save)
        else:
            if (self.i_inj.ndim > 1):
                for i in range(self.i_inj.shape[1]):
                    plots_results(self.neuron, self.t, self.i_inj[:,i], np.array(X[:,:,i]), suffix='target_%s%s' % (suffix,i), show=show,
                                  save=save)
            else:
                plots_results(self.neuron, self.t, self.i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save)

        if (dump):
            todump = [self.t, self.i_inj, X[:, V_pos], X[:, Ca_pos]]
            with open(data.DUMP_FILE, 'wb') as f:
                pickle.dump(todump, f)
            return data.DUMP_FILE



if __name__ == '__main__':
    i_injs = np.stack([params.i_inj_train, params.i_inj_train2], axis=1)
    t,i = params.give_train()
    param = data.get_vars('Integcomp_alternate_yolo-YAYYY')
    param = dict([(var, val[31]) for var, val in param.items()])
    for n in range(i.shape[1]):
        sim = HH_simul(t=t, i_inj=i[:,n])
        sim.comp_targ(param, params.DEFAULT)
    # sim.simul(show=True, save=False, dump=True)
    #
    # exit(0)

