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

from . import utils, config
from .neuron import BioNeuronTf, BioNeuronFix
from .optimize import Optimized


class Circuit:

    """Circuit of neurons with synapses"""
    def __init__(self, conns, neurons, tensors=False):
        self._tensors = tensors
        self._neurons = neurons
        if isinstance(conns, list):
            self._num = len(conns)
            inits_p = []
            variables = list(conns[0].values())[0].keys()
            for c in conns:
                # build dict for each circuit
                inits_p.append({var: np.array([syn[var] for syn in c.values()], dtype=np.float32) for var in variables})
            # merge them all in a new dimension
            self.init_p = {var: np.stack([mod[var] for mod in inits_p], axis=1) for var in variables}
            neurons.parallelize(self._num)
            self._connections = list(conns[0].keys())
        else:
            self._num = 1
            self.init_p = {var : np.array([p[var] for p in conns.values()], dtype=np.float32) for var in
                 list(conns.values())[0].keys()}
            self._connections = conns.keys()
        self._init_state = self._neurons.init_state
        self.dt = self._neurons.dt
        self._param = self.init_p
        syns = list(zip(*[k for k in self._connections]))
        self._pres = np.array(syns[0], dtype=np.int32)
        self._posts = np.array(syns[1], dtype=np.int32)
        self._syns = ['%s-%s' % (a,b) for a,b in zip(self._pres, self._posts)]
        self.n_synapse = len(self._pres)
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

    def step(self, hprev, curs, i=None):
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

            # if use while loop
            if i is not None:
                k = curs
                curs = curs[i]
                i += 1
            try:
                print('begin step')
                hprevs, extra = hprev
                hprev = hprevs[-1]
                print(hprev)
                for t in extra[0]:
                    print(t)
                # print('hprev in step ', hprev, 'input cur : ', curs)
                # print('extras : ', extra)
            except:
                extra = None
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
            try:
                h = self._neurons.step(hprev, final_curs)
            except:
                h = self._neurons.step(hprev, extra, final_curs)
                print('end step')
                hn, extra = h
                hgather = tf.concat([hprevs, [hn]], axis=0)
            return (hgather, extra), k, i
        else:
            # update neurons
            h = self._neurons.step(hprev, curs)
            # update synapses
            vpres = h[0, self._pres]
            vposts = h[0, self._posts]
            curs_syn = self.syn_curr(vpres, vposts)
            curs_post = np.zeros(curs.shape)
            for i in range(self._neurons.num):
                if i not in self._posts:
                    curs_post[i] = 0
                    continue
                curs_post[i] = np.sum(curs_syn[self._posts == i])
            return h, curs_post



    plot_output = config.NEURON_MODEL.plot_output


class CircuitTf(Circuit, Optimized):
    """
    Circuit using tensorflow
    """

    def __init__(self, inits_p=None, conns=None, neurons=None, fixed='all', constraints_n=None, dt=0.1):
        """

        Args:
            conns(dict): initial parameters for the synapses
            neurons(NeuronModel): if not None, all other parameters except conns are ignores
            inits_p(list of dict): initial parameters for the contained neurons
            fixed(list): fixed parameters for the neurons
            constraints_n(dict): constraints for the neurons
            dt(float): time step
        """
        if neurons is None:
            neurons = BioNeuronTf(inits_p, fixed=fixed, constraints=constraints_n, dt=dt)
        Optimized.__init__(self, dt=neurons.dt)
        Circuit.__init__(self, conns=conns, tensors=True, neurons=neurons)
        if self._num > 1:
            constraints_dic = give_constraints_syn(conns[0])
            #update constraint size for parallelization
            self.constraints_dic = dict(
                [(var, np.stack([val for _ in range(self._num)], axis=val.ndim)) if val.ndim > 1 else (var, val) for var, val in
                 constraints_dic.items()])
        else:
            self.constraints_dic = give_constraints_syn(conns)

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

    def build_graph(self, batch=None):
        self.reset()
        xshape = [None]
        initializer = self._init_state
        if batch is not None:
            xshape.append(None)
            # print('init shape :', initializer.shape)
            initializer = np.stack([initializer for _ in range(batch)], axis=1)
            self._neurons.init(batch)
            initializer = (np.stack([initializer ** ]), [self._neurons._neurons[-1].vstate])
        xshape.append(self._neurons.num)
        if self._num > 1:
            xshape.append(self._num)
            # if self.parallel is not None:
            #     xshape.append(self.parallel)
        curs_ = tf.placeholder(shape=xshape, dtype=tf.float32, name='input_current')

        i = tf.constant(0)
        cond = lambda hprev, curs_, i: tf.less(i, tf.shape(curs_)[0])
        res = tf.while_loop(cond,
                            body=self.step,
                            loop_vars=[initializer, curs_, i])
        for r in res:
            print(r)
        res = tf.stack([r[0] for r in res], axis=0)
        print('res', res)

        # res = tf.scan(self.step,
        #               curs_,
        #               initializer=initializer)

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

    def __init__(self, inits_p, conns, dt=0.1):
        neurons = BioNeuronFix(inits_p, dt=dt)
        Circuit.__init__(self, conns=conns, tensors=False, neurons=neurons)
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