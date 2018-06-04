import numpy as np
from HH_simul import HH_simul
import scipy as sp
import params
from utils import plots_output_mult

class Circuit():

    """
    neurons : objects to optimize

    """
    def __init__(self, neurons, conns, i_injs, t, i_out, init_p=params.SYNAPSE, dt=0.1):
        assert(len(neurons) == i_injs.shape[0])
        self.param = init_p
        self.neurons = neurons
        self.connections = conns
        self.t = t
        self.i_injs = i_injs
        self.i_out = i_out
        self.dt = dt
        self.param = {}
        for pre, post in conns:
            for k, v in init_p.items():
                self.param['%s-%s__%s'%(pre, post, k)] = v
        print(self.param)


    def syn_curr(self, syn, vprev, vpost):
        g = self.param['%s__G'%syn] / (1 + sp.exp((self.param['%s__mdp'%syn] - vprev)/self.param['%s__scale'%syn]))
        i = g*(self.param['%s__E'%syn] - vpost)
        return i


    def run_sim(self):
        self.simuls = []
        states = dict([ (i, np.zeros((len(self.neurons[0].init_state), len(self.t))) ) for i,n in enumerate(self.neurons)])
        for n in self.neurons:
            self.simuls.append(HH_simul(init_p=n.init_p))
        curs = np.zeros(len(self.simuls))
        for t in range(self.i_injs.shape[1]):
            curs = self.step(self.i_injs[:,t] + curs)

            for k,v in states.items():
                states[k][:,t] = self.simuls[k].state

        plots_output_mult(self.t, self.i_injs, [states[0][0,:], states[1][0,:]], [states[0][-1,:], states[1][-1,:]])



    def step(self, curs):
        next_curs = np.zeros(len(curs))
        #update neurons
        for i, n in enumerate(self.simuls):
            n.step(curs[i])
        #update synapses
        for pre, post in self.connections:
            vprev = self.simuls[pre].state[0]
            vpost = self.simuls[post].state[0]
            next_curs[post] += self.syn_curr('%s-%s'%(pre,post), vprev, vpost)
        return next_curs


    def optimize(self):
        pass