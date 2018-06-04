import scipy as sp
from Hodghux import HodgkinHuxley
import time
from utils import plots_results, plots_ik_from_v, plots_ica_from_v
import utils
import data
import params
import numpy as np
import pickle


class HH_simul(HodgkinHuxley):
    """Full Hodgkin-Huxley Model implemented in Python"""


    def __init__(self, init_p=params.DEFAULT, t=params.t, i_inj=params.i_inj, loop_func=None):
        HodgkinHuxley.__init__(self, init_p, tensors=False, loop_func=loop_func)
        self.t = t
        self.i_inj = i_inj
        self.dt = t[1] - t[0]
        self.state = self.init_state

    def get_volt(self):
        return self.state[0]

    def step(self, i):
        self.state = self.loop_func(self.state, i, self.dt, self)

    def calculate(self):
        X = []
        for i in self.i_inj:
            self.step(i)
            X.append(self.state)
        return np.array(X)

    """Compare different parameter sets on the same experiment"""
    def comp(self, ps):
        Vs = []
        Cacs = []
        for p in ps:
            self.param = p
            X = self.calculate()
            Vs.append(X[:,0])
            Cacs.append(X[:,-1])
        utils.plots_output_mult(self.t, self.i_inj, Vs, Cacs, save=False, show=True)

    """Compare 2 parameters sets"""
    def comp_targ(self, p, p_targ):
        self.param = p
        S = self.calculate()

        self.param = p_targ
        S_targ = self.calculate()

        utils.plots_output_double(self.t, self.i_inj, S[:,0], S_targ[:,0], S[:,-1], S_targ[:,-1], save=False, show=True)


    def Main(self, dump=False, sufix='', show=False, save=True):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        start = time.time()

        X = self.calculate()

        todump = np.vstack((self.t, self.i_inj, X[:, 0], X[:, -1]))

        print(time.time() - start)

        if (self.loop_func == self.ica_from_v):
            plots_ica_from_v(self.t, self.i_inj, np.array(X), suffix='target_%s' % sufix, show=show, save=save)
        elif (self.loop_func == self.ik_from_v):
            plots_ik_from_v(self.t, self.i_inj, np.array(X), suffix='target_%s' % sufix, show=show, save=save)
        else:
            plots_results(self, self.t, self.i_inj, np.array(X), suffix='target_%s' % sufix, show=show, save=save)

        if (dump):
            with open(data.DUMP_FILE, 'wb') as f:
                pickle.dump(todump, f)



if __name__ == '__main__':
    sim = HH_simul(t=params.t_train, i_inj=params.i_inj_train)
    sim.comp([params.PARAMS_RAND,params.DEFAULT])
    #
    # exit(0)

