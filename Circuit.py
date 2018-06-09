import numpy as np
from Neuron import Neuron_tf, HodgkinHuxley, Neuron_fix
import params
import scipy as sp
import tensorflow as tf

class Circuit():

    """
    Circuit of neurons with synapses

    """
    def __init__(self, conns, tensors=False):
        self.tensors = tensors
        self.connections = conns
        syns = zip(*[k for k in conns.iterkeys()])
        self.pres = np.array(syns[0], dtype=np.int32)
        self.posts = np.array(syns[1], dtype=np.int32)
        self.syns = ['%s-%s' % (a,b) for a,b in zip(self.pres, self.posts)]
        self.param = {}
        for k in self.connections.values()[0].keys():
            self.param[k] = [p[k] for n, p in self.connections.items()]

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
            idx_pres = np.vstack((np.zeros(self.pres.shape, dtype=np.int32), self.pres)).transpose()
            idx_post = np.vstack((np.zeros(self.pres.shape, dtype=np.int32), self.posts)).transpose()
            vpres = tf.gather_nd(hprev, idx_pres)
            vposts = tf.gather_nd(hprev, idx_post)
            curs_syn = self.syn_curr(vpres, vposts)
            curs_post = []
            for i in range(self.neurons.num):
                if i not in self.posts:
                    curs_post.append(0.)
                    continue
                curs_post.append(tf.reduce_sum(tf.gather(curs_syn, np.argwhere(self.posts == i))))
            # update neurons
            h = self.neurons.step(hprev, curs + tf.stack(curs_post))
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

    def __init__(self, inits_p, conns, loop_func=HodgkinHuxley.loop_func, fixed=params.ALL,
                 constraints_s=params.CONSTRAINTS_syn, constraints_n=params.CONSTRAINTS, dt=0.1):
        Circuit.__init__(self, conns=conns, tensors=True)
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
        self.neurons = Neuron_fix(inits_p, loop_func=loop_func, dt=dt)