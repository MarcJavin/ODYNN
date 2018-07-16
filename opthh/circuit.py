"""
.. module:: circuit
    :synopsis: Module containing implementation of neural circuits objects

.. moduleauthor:: Marc Javin
"""
import random

import numpy as np
import scipy as sp
import tensorflow as tf
import pylab as plt

from . import utils, config_model
from .neuron import BioNeuronTf, BioNeuronFix
from .optimize import Optimized

SYNAPSE1 = {
    'G': 1.,
    'mdp': -30.,
    'scale': 2.,
    'E': 20.
}
SYNAPSE2 = {
    'G': 10.,
    'mdp': 0.,
    'scale': 5.,
    'E': -10.
}
SYNAPSE = {
    'G': 5.,
    'mdp': -25.,
    'scale': 2.,
    'E': 0.
}
SYNAPSE_inhib = {
    'G': 1.,
    'mdp': -35.,
    'scale': -2.,
    'E': 20.
}
GAP = {'G_gap': 1.}

def give_constraints(conns):
    return {**give_constraints_syn(conns), **give_constraints_gap()}

def give_constraints_gap():
    return {'G_gap': np.array([1e-5, np.infty])}

def give_constraints_syn(conns):
    """constraints for synapse parameters

    Args:
      conns(dict): dictionnary of synapse parameters

    Returns:
        dict: constraints
    """
    scale_con = np.array([const_scale(True) if p['scale'] > 0 else const_scale(False) for p in conns.values()])
    return {'G': np.array([1e-5, np.infty]),
            'scale': scale_con.transpose()}


def const_scale(exc=True):
    if exc:
        return [1e-3, np.infty]
    else:
        return [-np.infty, -1e-3]


MAX_TAU = 200.
MIN_SCALE = 1.
MAX_SCALE = 50.
MIN_MDP = -40.
MAX_MDP = 30.
MAX_G = 10.


def get_syn_rand(exc=True):
    """Give random parameters dictionnary for a synapse

    Args:
      exc(bool): If True, give an excitatory synapse (Default value = True)

    Returns:
        dict: random parameters for a synapse
    """
    # scale is negative if inhibitory
    if exc:
        scale = random.uniform(MIN_SCALE, MAX_SCALE)
    else:
        scale = random.uniform(-MAX_SCALE, -MIN_SCALE)
    return {
        'G': random.uniform(0.01, MAX_G),
        'mdp': random.uniform(MIN_MDP, MAX_MDP),
        'scale': scale,
        'E': random.uniform(-20., 50.),
    }

VARS_SYN = list(SYNAPSE1.keys())
VARS_GAP = list(GAP.keys())

class Circuit:

    """Circuit of neurons with synapses"""

    def __init__(self, neurons, synapses={}, gaps={}, tensors=False):
        self._tensors = tensors
        self._neurons = neurons
        if isinstance(synapses, list) or isinstance(gaps, list):
            if gaps == {}:
                gaps = [{} for _ in range(len(synapses))]
                vars = VARS_SYN
            elif synapses == {}:
                synapses = [{} for _ in range(len(gaps))]
                vars = VARS_GAP
            else:
                vars = VARS_SYN + VARS_GAP
            if not isinstance(gaps, list) or not isinstance(synapses, list):
                raise AttributeError('Attributes conns and gaps should be of the same type')
            self._num = len(synapses)
            if len(gaps) != len(synapses):
                raise AttributeError('Attribute conns and gaps should have the same lengths, got {} and {}'.format(self._num, len(gaps)))
            inits_p = []
            for i in range(self._num):
                # build dict for each circuit
                init_p = {var: np.array([p[var] for p in synapses[i].values()], dtype=np.float32) for var in VARS_SYN}
                init_gap = {var: np.array([p[var] for p in gaps[i].values()], dtype=np.float32) for var in VARS_GAP}
                init_p.update(init_gap)
                inits_p.append(init_p)
            # merge them all in a new dimension
            self.init_p = {var: np.stack([mod[var] for mod in inits_p], axis=1) for var in vars}
            neurons.parallelize(self._num)
            self._connections = list(synapses[0].keys())
            self._gaps = list(gaps[0].keys())

        else:
            self._num = 1
            self.init_p = {var : np.array([p[var] for p in synapses.values()], dtype=np.float32) for var in VARS_SYN}
            init_gap = {var : np.array([p[var] for p in gaps.values()], dtype=np.float32) for var in VARS_GAP}
            self.init_p.update(init_gap)
            self._connections = synapses.keys()
            self._gaps = gaps.keys()
        self._init_state = self._neurons.init_state
        self.dt = self._neurons.dt
        self._param = self.init_p
        syns = list(zip(*[k for k in self._connections]))
        gaps_c = list(zip(*[k for k in self._gaps]))
        print('syn & gap', syns, gaps_c)
        if len(gaps_c)==0:
            gaps_c = [[], []]
        if len(syns)==0:
            syns = [[], []]
        print('syn & gap', syns, gaps_c)
        self._pres = np.hstack((syns[0], gaps_c[0], gaps_c[1])).astype(np.int32)
        self._posts = np.hstack((syns[1], gaps_c[1], gaps_c[0])).astype(np.int32)
        self.n_synapse = len(syns[0])
        self.n_gap = len(gaps_c[0])

        nb_neurons = len(np.unique(np.hstack((self._pres,self._posts))))
        if nb_neurons != self._neurons.num:
            raise AttributeError("Invalid number of neurons, got {}, expected {}".format(self._neurons.num, nb_neurons))

    @property
    def num(self):
        """Number of circuits contained in the object, used to train in parallel"""
        return self._num

    @property
    def neurons(self):
        """Neurons contained in the circuit"""
        return self._neurons

    @property
    def init_state(self):
        return self._init_state

    def gap_curr(self, vprev, vpost):
        G = self._param['G_gap']
        return G * (vpost - vprev)

    def syn_curr(self, vprev, vpost):
        """
        Compute the synaptic current

        Args:
          vprev(ndarray or tf.Tensor): presynaptic voltages
          vpost(ndarray or tf.Tensor): postsynaptic voltages

        Returns:
            ndarray of tf.Tensor: synaptic currents

        """
        G = self._param['G']
        mdp = self._param['mdp']
        scale = self._param['scale']
        if self._tensors:
            g = G * tf.sigmoid((vprev - mdp) / scale)
        else:
            g = G / (1 + sp.exp((mdp - vprev) / scale))
        return g * (self._param['E'] - vpost)

    def inter_curr(self, vprev, vpost):
        if(self.n_synapse == 0):
            return self.gap_curr(vprev, vpost)
        elif(self.n_gap == 0):
            return self.syn_curr(vprev, vpost)
        syns = self.syn_curr(vprev[:self.n_synapse], vpost[:,self.n_synapse])
        gaps = self.gap_curr(vprev[self.n_synapse:], vpost[self.n_synapse:])
        print(syns.shape)
        if self._tensors:
            return tf.concat([syns, gaps])
        else:
            return np.concatenate((syns, gaps))

    def step(self, hprev, curs):
        """run one time step

        For tensor :


        Args:
          hprev(ndarray or tf.Tensor): previous state vector
          curs(ndarray or tf.Tensor): input currents

        Returns:
            ndarray or tf.Tensor: updated state vector
        """
        if self._tensors:
            # update synapses
            #curs : [batch, neuron(, model)]
            #hprev : [state, batch, neuron(, model)] -> [state, neuron, batch(, model)]

            # if use extra init parameters
            try:
                hprev, extra = hprev
            except:
                extra = None
            ndim = 3 if self._num == 1 else 4
            perm_h = [0, 2, 1, 3][:ndim]
            print(perm_h, 'perm_h')
            print(self._num)
            perm_v = [1, 0, 2][:ndim-1]

            hprev_swap = tf.transpose(hprev, perm_h)
            idx_pres = np.stack((np.zeros(self._pres.shape, dtype=np.int32), self._pres), axis=1)
            idx_post = np.stack((np.zeros(self._pres.shape, dtype=np.int32), self._posts), axis=1)
            #[neuron, batch(, model)] -> [batch, neuron(, model)]

            vpres = tf.transpose(tf.gather_nd(hprev_swap, idx_pres), perm=perm_v)
            vposts = tf.transpose(tf.gather_nd(hprev_swap, idx_post), perm=perm_v)

            #voltage of the presynaptic cells
            curs_intern = self.inter_curr(vpres, vposts)
            # [batch, neuron(, model)] -> [neuron, batch(, model)]
            curs_intern = tf.transpose(curs_intern, perm=perm_v)
            curs_post = []
            for i in range(self._neurons.num):
                # 0 synaptic current if no synapse coming in
                if i in self._posts:
                    # [batch(, model)]
                    current_in = tf.reduce_sum(tf.gather_nd(curs_intern, np.argwhere(self._posts == i)), axis=0)
                else:
                    current_in = tf.reduce_sum(tf.zeros(tf.shape(curs)), axis=1)
                curs_post.append(current_in)
            final_curs = tf.add_n([tf.stack(curs_post, axis=1), curs])
            try:
                h = self._neurons.step(hprev, final_curs)
            except:
                h = self._neurons.step(hprev, extra, final_curs)
            return h
        else:
            # update neurons
            h = self._neurons.step(hprev, curs)
            # update synapses
            vpres = h[0, self._pres]
            vposts = h[0, self._posts]
            curs_intern = self.syn_curr(vpres, vposts)
            curs_post = np.zeros(curs.shape)
            for i in range(self._neurons.num):
                if i not in self._posts:
                    curs_post[i] = 0
                    continue
                curs_post[i] = np.sum(curs_intern[self._posts == i])
            return h, curs_post


    plot_output = config_model.NEURON_MODEL.plot_output


class CircuitTf(Circuit, Optimized):
    """
    Circuit using tensorflow
    """

    def __init__(self, neurons, synapses={}, gaps={}, tensors=False):
        """

        Args:
            synapses(dict): initial parameters for the synapses
            neurons(NeuronModel): if not None, all other parameters except conns are ignores
        """
        Optimized.__init__(self, dt=neurons.dt)
        Circuit.__init__(self, neurons=neurons, synapses=synapses, gaps=gaps, tensors=True)
        if self._num > 1:
            constraints_dic = give_constraints(synapses[0])
            #update constraint size for parallelization
            self.constraints_dic = dict(
                [(var, np.stack([val for _ in range(self._num)], axis=val.ndim)) if val.ndim > 1 else (var, val) for var, val in
                 constraints_dic.items()])
        else:
            self.constraints_dic = give_constraints(synapses)

    def parallelize(self, n):
        """Add a dimension of size n in the parameters

        Args:
          n: size of the new dimension
        """
        self.init_p = dict([(var, np.stack([val for _ in range(n)], axis=val.ndim)) for var, val in self.init_p.items()])
        self._neurons.parallelize(n)

    def reset(self):
        """prepare the variables as tensors, prepare the constraints, call reset for self._neurons"""
        self._param = {}
        self.constraints = []
        for var, val in self.init_p.items():
            self._param[var] = tf.get_variable(var, initializer=val, dtype=tf.float32)
            if var in self.constraints_dic:
                # add dimension for later
                con = self.constraints_dic[var]
                self.constraints.append(tf.assign(self._param[var], tf.clip_by_value(self._param[var], con[0], con[1])))
        self._neurons.reset()
        self._init_state = self._neurons.init_state

    def build_graph(self, batch=1):
        self.reset()
        xshape = [None]
        initializer = self._init_state.astype(np.float32)
        if batch != 1:
            xshape.append(None)
            initializer = np.stack([initializer for _ in range(batch)], axis=1)
        self._neurons.init(batch)
        xshape.append(self._neurons.num)
        print("num neurons : ", self._neurons.num)
        if self._num > 1:
            xshape.append(self._num)
            # if self.parallel is not None:
            #     xshape.append(self.parallel)
        extra_state = self._neurons.hidden_init_state
        curs_ = tf.placeholder(shape=xshape, dtype=tf.float32, name='input_current')
        infer_shape = True
        if extra_state is not None:
            initializer = (initializer, extra_state)
            infer_shape = False

        res = tf.scan(self.step,
                      curs_,
                      infer_shape=infer_shape,
                      initializer=initializer)

        if extra_state is not None:
            res = res[0]

        return curs_, res


    def calculate(self, i):
        """
        Iterate over i (current) and return the state variables obtained after each step

        Args:
          i(ndarray): input current

        Returns:
            ndarray: state vectors concatenated [i.shape[0], len(self.init_state)(, i.shape[1]), self.num]
        """
        if i.ndim > 1 and self._num == 1 or i.ndim > 2 and self._num > 1:
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

    def apply_init(self, session):
        self._neurons.apply_init(session)

    def get_params(self):
        return self._param.items()

    def study_vars(self, p):
        self.plot_vars(p, func=utils.bar, suffix='compared', show=False, save=True)
        self.plot_vars(p, func=utils.box, suffix='boxes', show=False, save=True)

    def plot_vars(self, var_dic, suffix="", show=True, save=False, func=utils.plot):
        """plot variation/comparison/boxplots of synaptic variables

        Args:
          var_dic(dict): synaptic parameters, each value of size [time, n_synapse, parallelization]
          suffix:  (Default value = "")
          show(bool): If True, show the figure (Default value = True)
          save(bool): If True, save the figure (Default value = False)
          func:  (Default value = plot)
        """

        def oneplot(var_d, name):
            labels = ['G', 'mdp', 'E', 'scale']
            if func == utils.box:
                func(var_d, utils.COLORS[:len(labels)], labels)
            else:
                for i, var in enumerate(labels):
                    plt.subplot(2, 2, i + 1)
                    func(plt, var_d[var])
                    plt.ylabel(var)
            plt.tight_layout()
            utils.save_show(show, save, name='{}_{}'.format(name, suffix), dpi=300)
            plt.close()

        if (self._num > 1):
            # if parallelization, compare run on each synapse
            for i in range(var_dic['E'].shape[0]):
                var_d = {var: val[i] for var, val in var_dic.items()}
                oneplot(var_d, 'Synapse_{}'.format(i))
        else:
            # if not, compare all synapses together
            oneplot(var_dic, 'All_Synapses')
    

class CircuitFix(Circuit):

    def __init__(self, neurons, synapses={}, gaps={}):
        Circuit.__init__(self, neurons=neurons, synapses=synapses, gaps=gaps, tensors=False)
        self._param = self.init_p

    def calculate(self, i_inj):
        """
        Simulate the circuit with a given input current.

        Args:
            i_inj(ndarray): input current

        Returns:
            ndarray, ndarray: state vector and synaptical currents
        """
        states = []#np.zeros((np.hstack((len(i_inj), self.neurons.init_state.shape))))
        curs = []#np.zeros(i_inj.shape)
        h = self._neurons.init_state

        for t in range(len(i_inj)):
            if t == 0:
                h, c = self.step(h, curs=i_inj[t])
            else:
                h, c = self.step(h, curs=i_inj[t] + curs[t - 1])
            curs.append(c)
            states.append(h)
        return np.stack(states, axis=0), np.stack(curs, axis=0)


