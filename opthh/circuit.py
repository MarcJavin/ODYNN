"""
.. module:: circuit
    :synopsis: Module containing implementation of neural circuits objects

.. moduleauthor:: Marc Javin
"""

import numpy as np
import scipy as sp
import tensorflow as tf

import neuron_params, params, utils
from Neuron import NeuronTf, NeuronFix
from optimize import Optimized


class Circuit:

    """
    Circuit of neurons with synapses

    """
    def __init__(self, conns, neurons, tensors=False):
        self._tensors = tensors
        self._neurons = neurons
        if isinstance(conns, list):
            self.num = len(conns)
            inits_p = []
            variables = list(conns[0].values())[0].keys()
            for c in conns:
                # build dict for each circuit
                inits_p.append(dict(
                    [(var, np.array([syn[var] for syn in c.values()], dtype=np.float32)) for var in variables]))
            # merge them all in a new dimension
            self.init_p = dict([(var, np.stack([mod[var] for mod in inits_p], axis=1)) for var in variables])
            neurons.parallelize(self.num)
            self._connections = list(conns[0].keys())
        else:
            self.num = 1
            self.init_p = dict(
                [(var, np.array([p[var] for p in conns.values()], dtype=np.float32)) for var in
                 list(conns.values())[0].keys()])
            self._connections = conns.keys()
        self.init_state = self._neurons.init_state
        self.dt = self._neurons.dt
        self._param = self.init_p
        syns = list(zip(*[k for k in self._connections]))
        self._pres = np.array(syns[0], dtype=np.int32)
        self._posts = np.array(syns[1], dtype=np.int32)
        self._syns = ['%s-%s' % (a,b) for a,b in zip(self._pres, self._posts)]
        self.n_synapse = len(self._pres)
        assert(len(np.unique(np.hstack((self._pres,self._posts)))) == self._neurons.num), "Invalid number of neurons"

    def syn_curr(self, vprev, vpost):
        """synaptic current"""
        G = self._param['G']
        mdp = self._param['mdp']
        scale = self._param['scale']
        if self._tensors:
            g = G * tf.sigmoid((vprev - mdp) / scale)
        else:
            g = G / (1 + sp.exp((mdp - vprev) / scale))
        return g * (self._param['E'] - vpost)

    def step(self, hprev, curs):
        """run one time step"""
        if self._tensors:
            # update synapses
            #curs : [batch, neuron(, model)]
            #hprev : [state, batch, neuron(, model)] -> [state, neuron, batch(, model)]
            try:
                hprev_swap = tf.transpose(hprev, [0,2,1])
            except:
                hprev_swap = tf.transpose(hprev, [0, 2, 1, 3])
            idx_pres = np.stack((np.zeros(self._pres.shape, dtype=np.int32), self._pres), axis=1)#transpose()
            idx_post = np.stack((np.zeros(self._pres.shape, dtype=np.int32), self._posts), axis=1)#.transpose()
            #[neuron, batch(, model)] -> [batch, neuron(, model)]
            try:
                vpres = tf.transpose(tf.gather_nd(hprev_swap, idx_pres), perm=[1,0])
                vposts = tf.transpose(tf.gather_nd(hprev_swap, idx_post), perm=[1,0])
            except:
                vpres = tf.transpose(tf.gather_nd(hprev_swap, idx_pres), perm=[1, 0, 2])
                vposts = tf.transpose(tf.gather_nd(hprev_swap, idx_post), perm=[1, 0, 2])
            #voltage of the presynaptic cells
            curs_syn = self.syn_curr(vpres, vposts)
            # [batch, neuron(, model)] -> [neuron, batch(, model)]
            try:
                curs_syn = tf.transpose(curs_syn, perm=[1,0])
            except:
                curs_syn = tf.transpose(curs_syn, perm=[1,0,2])
            curs_post = []
            for i in range(self._neurons.num):
                if i not in self._posts:
                    #0 synaptic current if no synapse coming in
                    curs_post.append(tf.reduce_sum(tf.zeros(tf.shape(curs)), axis=1))
                    continue
                #[batch(, model)]
                curs_post.append(tf.reduce_sum(tf.gather_nd(curs_syn, np.argwhere(self._posts == i)), axis=0))
            final_curs = tf.add_n([tf.stack(curs_post, axis=1), curs])
            h = self._neurons.step(hprev, final_curs)
            return h
        else:
            # update neurons
            self._neurons.step(curs)
            # update synapses
            vpres = self._neurons.state[0, self._pres]
            vposts = self._neurons.state[0, self._posts]
            curs_syn = self.syn_curr(vpres, vposts)
            curs_post = np.zeros(curs.shape)
            for i in range(self._neurons.num):
                if i not in self._posts:
                    curs_post[i] = 0
                    continue
                curs_post[i] = np.sum(curs_syn[self._posts == i])
            return curs_post

    @staticmethod
    def plot_vars(*args, **kwargs):
        return utils.plot_vars_syn(*args, **kwargs)


class CircuitTf(Circuit, Optimized):

    def __init__(self, inits_p, conns, fixed=neuron_params.ALL, constraints_n=neuron_params.CONSTRAINTS, dt=0.1):
        neurons = NeuronTf(inits_p, fixed=fixed, constraints=constraints_n, dt=dt)
        Circuit.__init__(self, conns=conns, tensors=True, neurons=neurons)
        Optimized.__init__(self)
        if self.num > 1:
            constraints_dic = params.give_constraints_syn(conns[0])
            #update constraint size for parallelization
            self.constraints_dic = dict(
                [(var, np.stack([val for _ in range(self.num)], axis=val.ndim)) if val.ndim > 1 else (var, val) for var, val in
                 constraints_dic.items()])
        else:
            self.constraints_dic = params.give_constraints_syn(conns)

    def parallelize(self, n):
        """Add a dimension of size n in the parameters"""
        self.init_p = dict([(var, np.stack([val for _ in range(n)], axis=val.ndim)) for var, val in self.init_p.items()])
        self._neurons.parallelize(n)

    def reset(self):
        """build tf graph"""
        self._param = {}
        self.constraints = []
        for var, val in self.init_p.items():
            self._param[var] = tf.get_variable(var, initializer=val, dtype=tf.float32)
            if var in self.constraints_dic:
                # add dimension for later
                con = self.constraints_dic[var]
                self.constraints.append(tf.assign(self._param[var], tf.clip_by_value(self._param[var], con[0], con[1])))
        self._neurons._reset()

    def build_graph(self, batch=None):
        self.reset()
        xshape = [None]
        initializer = self.init_state
        print(initializer.shape)
        if batch:
            xshape.append(None)
            initializer = np.stack([initializer for _ in range(batch)], axis=1)
        xshape.append(self._neurons.num)
        if self.num > 1:
            xshape.append(self.num)
            # if self.parallel is not None:
            #     xshape.append(self.parallel)
        curs_ = tf.placeholder(shape=xshape, dtype=tf.float32, name='input_current')

        print(curs_.shape)
        print(initializer.shape)
        res = tf.scan(self.step,
                      curs_,
                      initializer=initializer.astype(np.float32))

        return curs_, res

    def calculate(self, i):
        if i.ndim > 1 and self.num == 1 or i.ndim > 2 and self.num > 1:
            input_cur, res_ = self.build_graph(batch=i.shape[i.ndim-2])
        else:
            input_cur, res_ = self.build_graph()
        with tf.Session() as sess:
            results = sess.run(res_, feed_dict={
                input_cur: i
            })
        return results
    
    def settings(self):
        return ('Circuit optimization'.center(20, '.') + '\n' +
                'Connections : \n %s \n %s' % (self._pres, self._posts) + '\n' +
                'Initial synaptic params : %s' % self._connections + '\n' +
                self._neurons.settings())

    def apply_constraints(self, session):
        session.run(self.constraints)
        self._neurons.apply_constraints(session)

    def get_params(self):
        return self._param.items()
    

class CircuitFix(Circuit):

    def __init__(self, inits_p, conns, dt=0.1):
        neurons = NeuronFix(inits_p, dt=dt)
        Circuit.__init__(self, conns=conns, tensors=False, neurons=neurons)
        self._param = self.init_p