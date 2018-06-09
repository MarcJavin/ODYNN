import numpy as np
from Hodghux import HodgkinHuxley
from Circuit import Circuit_fix
from utils import plots_output_mult
from data import DUMP_FILE
import pickle


class Circuit_simul():
    """
    neurons : objects to optimize

    """

    def __init__(self, inits_p, conns, t, i_injs, loop_func=HodgkinHuxley.loop_func, dt=0.1):
        assert(len(inits_p) == i_injs.shape[1])
        self.circuit = Circuit_fix(inits_p=inits_p, conns=conns, loop_func=loop_func, dt=dt)
        self.connections = conns
        self.t = t
        self.i_injs = i_injs


    """runs the entire simulation"""

    def run_sim(self, dump=False, general=True, show=False, save=True):
        #[state, neuron, time]
        states = np.zeros((np.hstack((len(self.t), self.circuit.neurons.init_state.shape))))
        curs = np.zeros(self.i_injs.shape)

        for t in range(len(self.t)):
            if (t == 0):
                curs[t, :] = self.circuit.step(curs=self.i_injs[t ,:])
            else:
                curs[t, :] = self.circuit.step(curs=self.i_injs[t, :] + curs[t - 1, :])

            states[t, :, :] = self.circuit.neurons.state

        plots_output_mult(self.t, self.i_injs, states[:,0,:], states[:,-1,:],
                          i_syn=curs, show=show, save=save, suffix='TARGET')

        if (dump):
            for i in range(self.circuit.neurons.num):
                if(general):
                    cur = self.i_injs
                else:
                    cur = self.i_injs[:, i] + curs[:, i]

                todump = [self.t, cur, states[:,0,i], states[:,-1,i]]
                with open(DUMP_FILE + str(i), 'wb') as f:
                    pickle.dump(todump, f)
            return DUMP_FILE
