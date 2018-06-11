import numpy as np
from Neuron import Neuron_tf, HodgkinHuxley, Neuron_fix
import params
import scipy as sp
import tensorflow as tf
import copy

class Circuit():

    """
    Circuit of neurons with synapses

    """
    def __init__(self, conns, tensors=False):
        self.tensors = tensors
        self.connections = conns
        syns = zip(*[k for k in conns.keys()])
        self.pres = np.array(syns[0], dtype=np.int32)
        self.num = len(self.pres)
        self.posts = np.array(syns[1], dtype=np.int32)
        self.syns = ['%s-%s' % (a,b) for a,b in zip(self.pres, self.posts)]
        self.init_p = {}
        for k in self.connections.values()[0].keys():
            self.init_p[k] = [p[k] for n, p in self.connections.items()]

    """synaptic current"""
    def syn_curr(self, vprev, vpost):
        G = self.param['G']
        mdp = self.param['mdp']
        scale = self.param['scale']
        if(self.tensors):
            g = G * tf.sigmoid((vprev - mdp) / scale)
        else:
            g = G / (1 + sp.exp((mdp - vprev) / scale))
        return g * (self.param['E'] - vpost)

    """run one time step"""
    def step(self, hprev=None, curs=[]):
        if(self.tensors):
            # update synapses
            #swap for synapses
            hprev_swap = tf.transpose(hprev, [0,2,1])
            idx_pres = np.vstack((np.zeros(self.pres.shape, dtype=np.int32), self.pres)).transpose()
            idx_post = np.vstack((np.zeros(self.pres.shape, dtype=np.int32), self.posts)).transpose()
            vpres = tf.gather_nd(hprev_swap, idx_pres)
            vposts = tf.gather_nd(hprev_swap, idx_post)
            #voltage of the presynaptic cells
            curs_syn = self.syn_curr(tf.transpose(vpres), tf.transpose(vposts))
            curs_syn = tf.transpose(curs_syn)
            curs_post = []
            for i in range(self.neurons.num):
                if i not in self.posts:
                    #0 synaptic current if no synapse coming in
                    curs_post.append((tf.zeros(tf.shape(curs)[0])))
                    continue
                curs_post.append(tf.reduce_sum(tf.gather_nd(curs_syn, np.argwhere(self.posts == i)), axis=0))
            final_curs = curs + tf.stack(curs_post, axis=1)
            h = self.neurons.step(hprev, final_curs)
            return h
        else:
            # update neurons
            self.neurons.step(curs)
            # update synapses
            vpres = self.neurons.state[0, self.pres]
            vposts = self.neurons.state[0, self.posts]
            curs_syn = self.syn_curr(vpres, vposts)
            curs_post = np.zeros(len(curs))
            for i in range(self.neurons.num):
                if i not in self.posts:
                    curs_post[i] = 0
                    continue
                curs_post[i] = np.sum(curs_syn[self.posts == i])
            return curs_post


class Circuit_tf(Circuit):

    def __init__(self, inits_p, conns, loop_func=HodgkinHuxley.loop_func, fixed=params.ALL, constraints_n=params.CONSTRAINTS, dt=0.1):
        Circuit.__init__(self, conns=conns, tensors=True)
        constraints_s = params.give_constraints_syn(conns)
        self.neurons = Neuron_tf(inits_p, loop_func=loop_func, fixed=fixed, constraints=constraints_n, dt=dt)
        self.constraints_dic = constraints_s

    """build tf graph"""
    def reset(self):
        self.param = {}
        self.constraints = []
        for var in self.connections.values()[0].keys():
            self.param[var] = tf.get_variable(var, initializer=[p[var] for n, p in self.connections.items()],
                                            dtype=tf.float32)
            if(var in self.constraints_dic):
                con = self.constraints_dic[var]
                self.constraints.append(tf.assign(self.param[var], tf.clip_by_value(self.param[var], con[0], con[1])))

class Circuit_fix(Circuit):

    def __init__(self, inits_p, conns, loop_func=HodgkinHuxley.loop_func, dt=0.1):
        Circuit.__init__(self, conns=conns, tensors=False)
        self.param = self.init_p
        self.neurons = Neuron_fix(inits_p, loop_func=loop_func, dt=dt)