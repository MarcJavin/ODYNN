import scipy as sp
from Hodghux import HodgkinHuxley
import time
from utils import plots_results, plots_ik_from_v, plots_ica_from_v
import utils
import params
import numpy as np
import pickle


class HH_simul(HodgkinHuxley):
    """Full Hodgkin-Huxley Model implemented in Python"""


    def __init__(self, init_p=params.DEFAULT, t=params.t, i_inj=params.i_inj):
        HodgkinHuxley.__init__(self, init_p, tensors=False)
        self.t = t
        self.i_inj = i_inj
        self.dt = t[1] - t[0]

    # def fitness(self, params):
    #     idx = 0
    #     for k, v in self.param.items():
    #         self.param[v] = params[idx]
    #         idx += 1
    #     print(self.param)
    #     S = [[-50, 0., 0.95, 0, 0, 1, 0]]
    #     div = DT/self.dt
    #     for i in X:
    #         S.append(self.integ_comp(S[-1], i, self.dt))
    #         s = S[-1]
    #         for d in range(div-1):
    #             s = self.integ_comp(s, i, self.dt)
    #     S = np.array(S[1:])
    #     V = S[:,0]
    #     mse = ((Y - V)**2).mean()
    #     return mse
    #
    # def get_bounds(self):
    #     m = -100
    #     M = 1000
    #     low = np.tile([0, m, m], 7), [0, 0, 0, 0, 0, m, m, m]
    #     up = np.full((29), M)
    #     return (low, up)

    def calculate(self):
        X = [self.get_init_state()]

        for i in self.i_inj:
            X.append(self.loop_func(X[-1], i, self.dt, self))
        return np.array(X[1:])

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
            with open(utils.DUMP_FILE, 'wb') as f:
                pickle.dump(todump, f)



if __name__ == '__main__':
    runner = HH_simul()
    runner.Main()

