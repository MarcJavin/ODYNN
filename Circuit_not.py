import numpy as np
from Hodghux import Neuron_set_fix, HodgkinHuxley
import scipy as sp
import params
from utils import plots_output_mult, set_dir, plot_loss_rate
from data import FILE_LV, DUMP_FILE
import pickle


class Circuit():
    """
    neurons : objects to optimize

    """

    def __init__(self, inits_p, conns, i_injs, t, loop_func=HodgkinHuxley.loop_func, i_out=None, dt=0.1, tensors=False):
        assert(len(inits_p) == i_injs.shape[0])
        self.neurons = Neuron_set_fix(inits_p, loop_func=loop_func, dt=dt)
        self.connections = conns
        self.t = t
        self.i_injs = i_injs
        self.i_out = i_out
        self.param = {}
        self.tensors = tensors
        for (pre, post), p in conns.items():
            for k, v in p.items():
                name = '%s-%s__%s' % (pre, post, k)
                self.param[name] = v

    """synaptic current"""

    def syn_curr(self, syn, vprev, vpost):
        g = self.param['%s__G' % syn] / (
                    1 + sp.exp((self.param['%s__mdp' % syn] - vprev) / self.param['%s__scale' % syn]))
        i = g * (self.param['%s__E' % syn] - vpost)
        return i


    """run one time step"""
    def step(self, curs):

        next_curs = np.zeros(len(curs))
        # update neurons
        self.neurons.step(curs)
        # update synapses
        for pre, post in self.connections.iterkeys():
            vprev = self.neurons.state[0,pre]
            vpost = self.neurons.state[0,post]
            next_curs[post] += self.syn_curr('%s-%s' % (pre, post), vprev, vpost)
        return next_curs

    """runs the entire simulation"""

    def run_sim(self, show=True, dump=False, general=True):
        #[state, neuron, time]
        states = np.zeros((np.hstack((self.neurons.init_state.shape, len(self.t)))))
        print(states.shape)
        curs = np.zeros(self.i_injs.shape)

        for t in range(self.i_injs.shape[1]):
            if (t == 0):
                curs[:, t] = self.step(self.i_injs[:, t])
            else:
                curs[:, t] = self.step(self.i_injs[:, t] + curs[:, t - 1])

            states[:, :, t] = self.neurons.state

        plots_output_mult(self.t, self.i_injs, [states[0,0, :], states[0,1, :]], [states[-1,0, :], states[-1,1, :]],
                          i_syn=curs, show=show)

        if (dump):
            for i in range(self.neurons.num):
                if(general):
                    cur = self.i_injs
                else:
                    cur = self.i_injs[i, :] + curs[i, :]

                todump = [self.t, cur, states[0,i, :], states[-1,i, :]]
                with open(DUMP_FILE + str(i), 'wb') as f:
                    pickle.dump(todump, f)
            return DUMP_FILE
