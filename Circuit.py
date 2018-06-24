import numpy as np
from Neuron import NeuronTf, HodgkinHuxley, NeuronFix
import neuron_params, params
import scipy as sp
import tensorflow as tf
import copy

class Circuit():

    """
    Circuit of neurons with synapses

    """
    def __init__(self, conns, neurons, tensors=False):
        self.tensors = tensors
        self.neurons = neurons
        if (isinstance(conns, list)):
            self.num = len(conns)
            inits_p = []
            variables = list(conns[0].values())[0].keys()
            for c in conns:
                #build dict for each circuit
                inits_p.append(dict(
                    [(var, np.array([syn[var] for syn in c.values()], dtype=np.float32)) for var in variables]))
            #merge them all in a new dimension
            self.init_p = dict([(var, np.stack([mod[var] for mod in inits_p], axis=1)) for var in variables])
            neurons.parallelize(self.num)
            self.connections = list(conns[0].keys())
        else:
            self.num = 1
            self.init_p = dict(
                [(var, np.array([p[var] for p in conns.values()], dtype=np.float32)) for var in
                 list(conns.values())[0].keys()])
            self.connections = conns.keys()
        self.init_state = self.neurons.init_state
        self.dt = self.neurons.dt
        self.param = self.init_p
        syns = list(zip(*[k for k in self.connections]))
        self.pres = np.array(syns[0], dtype=np.int32)
        self.posts = np.array(syns[1], dtype=np.int32)
        self.syns = ['%s-%s' % (a,b) for a,b in zip(self.pres, self.posts)]
        self.n_synapse = len(self.pres)
        assert(len(np.unique(np.hstack((self.pres,self.posts)))) == self.neurons.num), "Invalid number of neurons"

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
    def step(self, hprev, curs):
        if(self.tensors):
            # update synapses
            #curs : [batch, neuron(, model)]
            #hprev : [state, batch, neuron(, model)] -> [state, neuron, batch(, model)]
            try:
                hprev_swap = tf.transpose(hprev, [0,2,1])
            except:
                hprev_swap = tf.transpose(hprev, [0, 2, 1, 3])
            idx_pres = np.stack((np.zeros(self.pres.shape, dtype=np.int32), self.pres), axis=1)#transpose()
            idx_post = np.stack((np.zeros(self.pres.shape, dtype=np.int32), self.posts), axis=1)#.transpose()
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
            for i in range(self.neurons.num):
                if i not in self.posts:
                    #0 synaptic current if no synapse coming in
                    curs_post.append(tf.reduce_sum(tf.zeros(tf.shape(curs)), axis=1))
                    continue
                #[batch(, model)]
                curs_post.append(tf.reduce_sum(tf.gather_nd(curs_syn, np.argwhere(self.posts == i)), axis=0))
            final_curs = tf.add_n([tf.stack(curs_post, axis=1), curs])
            h = self.neurons.step(hprev, final_curs)
            return h
        else:
            # update neurons
            self.neurons.step(curs)
            # update synapses
            vpres = self.neurons.state[0, self.pres]
            vposts = self.neurons.state[0, self.posts]
            curs_syn = self.syn_curr(vpres, vposts)
            curs_post = np.zeros(curs.shape)
            for i in range(self.neurons.num):
                if i not in self.posts:
                    curs_post[i] = 0
                    continue
                curs_post[i] = np.sum(curs_syn[self.posts == i])
            return curs_post


class Circuit_tf(Circuit):

    def __init__(self, inits_p, conns, fixed=neuron_params.ALL, constraints_n=neuron_params.CONSTRAINTS, dt=0.1):
        neurons = NeuronTf(inits_p, fixed=fixed, constraints=constraints_n, dt=dt)
        Circuit.__init__(self, conns=conns, tensors=True, neurons=neurons)
        if(self.num > 1):
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
        self.neurons.parallelize(n)

    """build tf graph"""
    def reset(self):
        self.param = {}
        self.constraints = []
        for var, val in self.init_p.items():
            self.param[var] = tf.get_variable(var, initializer=val, dtype=tf.float32)
            if (var in self.constraints_dic):
                # add dimension for later
                con = self.constraints_dic[var]
                self.constraints.append(tf.assign(self.param[var], tf.clip_by_value(self.param[var], con[0], con[1])))
        self.neurons.reset()

    def build_graph(self, batch=None):
        self.reset()
        xshape = [None]
        initializer = self.init_state
        print(initializer.shape)
        if (batch):
            xshape.append(None)
            initializer = np.stack([initializer for _ in range(batch)], axis=1)
        xshape.append(self.neurons.num)
        if (self.num > 1):
            xshape.append(self.num)
            # if(self.parallel is not None):
            #     xshape.append(self.parallel)
        curs_ = tf.placeholder(shape=xshape, dtype=tf.float32, name='input_current')

        print(curs_.shape)
        print(initializer.shape)
        res = tf.scan(self.step,
                      curs_,
                      initializer=initializer.astype(np.float32))

        return curs_, res

    def calculate(self, i):
        if (i.ndim > 1 and self.num == 1 or i.ndim > 2 and self.num > 1):
            input_cur, res_ = self.build_graph(batch=i.shape[i.ndim-2])
        else:
            input_cur, res_ = self.build_graph()
        with tf.Session() as sess:
            results = sess.run(res_, feed_dict={
                input_cur: i
            })
        return results

class Circuit_fix(Circuit):

    def __init__(self, inits_p, conns, dt=0.1):
        neurons = NeuronFix(inits_p, dt=dt)
        Circuit.__init__(self, conns=conns, tensors=False, neurons=neurons)
        self.param = self.init_p