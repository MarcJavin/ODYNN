"""
.. module:: circuit
    :synopsis: Module containing implementation of neural circuits objects

.. moduleauthor:: Marc Javin
"""
import random

import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import pylab as plt
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle

from . import utils, model, neuron

from .optim import Optimized

SYNAPSE1 = {
    'G': 5.,
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
    'scale': 8.,
    'E': 0.
}
SYNAPSE_inhib = {
    'G': 2.,
    'mdp': -25.,
    'scale': 8.,
    'E': -70.
}
SYNAPSE_inhib2 = {
    'G': 3.,
    'mdp': -35.,
    'scale': -6.,
    'E': 20.
}
GAP = {'G_gap': 1.}

MAX_TAU = 200.
MIN_SCALE = 1.
MAX_SCALE = 50.
MIN_MDP = -40.
MAX_MDP = 30.
MIN_G = 1.e-6
MAX_G = 0.1

def give_constraints(conns):
    return {**give_constraints_syn(conns), **give_constraints_gap()}

def give_constraints_gap():
    return {'G_gap': np.array([1e-7, MAX_G])}

def give_constraints_syn(conns):
    """constraints for synapse parameters

    Args:
      conns(dict): dictionnary of synapse parameters

    Returns:
        dict: constraints
    """
    E_con = np.array([const_E(p['E'] > -60) for p in conns.values()]).transpose()
    return {'G': np.array([1e-7, MAX_G]),
            'scale': np.array([1e-3, np.infty]),
            'E' : E_con}


def const_E(exc=True):
    if exc:
        return [-70, np.infty]
    else:
        return [-np.infty, -50]


def get_syn_rand(exc=True):
    """Give random parameters dictionnary for a synapse

    Args:
      exc(bool): If True, give an excitatory synapse (Default value = True)

    Returns:
        dict: random parameters for a synapse
    """
    # scale is negative if inhibitory
    if exc:
        E = random.uniform(-60, 30)
    else:
        E = random.uniform(-100, -60)
    return {
        'G': random.uniform(MIN_G, MAX_G),
        'mdp': random.uniform(MIN_MDP, MAX_MDP),
        'scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'E': E
    }

def get_gap_rand():
    return {'G_gap' : random.uniform(0.01, MAX_G)}

VARS_SYN = list(SYNAPSE1.keys())
VARS_GAP = list(GAP.keys())

class Circuit:

    """Circuit of neurons with synapses and gap junctions"""

    def __init__(self, neurons, synapses={}, gaps={}, tensors=False, labels=None, sensors=set(), commands=set()):
        self._tensors = tensors
        self._neurons = neurons
        self.sensors = sensors
        self.commands = commands
        self.inter = set(list(range(self._neurons.num))) - self.sensors - self.commands
        self.labels = labels
        if self.labels is None:
            self.labels = {i:i for i in range(neurons.num)}
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
                init_gap = {var: np.tile(np.array([p[var] for p in gaps[i].values()], dtype=np.float32), 2) for var in VARS_GAP}
                init_p.update(init_gap)
                inits_p.append(init_p)
            # merge them all in a new dimension
            self._init_p = {var: np.stack([mod[var] for mod in inits_p], axis=1) for var in vars}
            neurons.parallelize(self._num)
            self.synapses = synapses[0]
            self.gaps = gaps[0]

        else:
            self._num = 1
            self._init_p = {var : np.array([p[var] for p in synapses.values()], dtype=np.float32) for var in VARS_SYN}
            init_gap = {var : np.tile(np.array([p[var] for p in gaps.values()], dtype=np.float32), 2) for var in VARS_GAP}
            self._init_p.update(init_gap)
            self.synapses = synapses
            self.gaps = gaps
        self._init_state = self._neurons.init_state
        self.dt = self._neurons.dt
        self._param = self._init_p
        syns = list(zip(*[k for k in self.synapses.keys()]))
        gaps_c = list(zip(*[k for k in self.gaps.keys()]))
        if len(gaps_c)==0:
            gaps_c = [[], []]
        if len(syns)==0:
            syns = [[], []]
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
        return G * (vprev - vpost)

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
        if self.num > 1:
            syns = self.syn_curr(vprev[...,:self.n_synapse,:], vpost[...,:self.n_synapse,:])
            gaps = self.gap_curr(vprev[...,self.n_synapse:,:], vpost[...,self.n_synapse:,:])
            axis = -2
        else:
            syns = self.syn_curr(vprev[...,:self.n_synapse], vpost[...,:self.n_synapse])
            gaps = self.gap_curr(vprev[...,self.n_synapse:], vpost[...,self.n_synapse:])
            axis = 1
        if self._tensors:
            return tf.concat([syns, gaps], axis=axis)
        else:
            return np.concatenate((syns, gaps), axis=axis)

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

            if h.ndim > 2:
                hs = np.swapaxes(h, 1, 2)
                vpres = np.swapaxes(hs[0, self._pres], 0, 1)
                vposts = np.swapaxes(hs[0, self._posts], 0, 1)
                curs_intern = np.swapaxes(self.inter_curr(vpres, vposts), 0, 1)
            else:
                # update synapses
                vpres = h[0, self._pres]
                vposts = h[0, self._posts]
                curs_intern = self.inter_curr(vpres, vposts)
            curs_post = np.zeros(curs.shape)
            for i in range(self._neurons.num):
                if i not in self._posts:
                    curs_post[...,i] = 0
                    continue
                curs_post[...,i] = np.sum(curs_intern[self._posts == i], axis=0)
            return h, curs_post

    def plot(self, show=True, save=False):
        """
        Plot the circuit using networkx
        Args:
            show(bool):
            save(bool:

        Returns:

        """
        G = nx.MultiDiGraph()
        exc = 'green'
        inh = 'Crimson'
        gap = 'Gold'
        synplot = [(k[0], k[1], {'color': inh}) if v['E'] < -60 else (k[0], k[1], {'color': exc}) for k, v in
                   self.synapses.items()]
        G.add_edges_from(synplot)
        G.add_edges_from([(k[0], k[1], {'color': gap}) for k in self.gaps.keys()])
        G.add_edges_from([(k[1], k[0], {'color': gap}) for k in self.gaps.keys()])
        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        pos = nx.layout.circular_layout(G)
        colors = []
        for i in range(self._neurons.num):
            edges = G.out_edges(i, data='color')
            cols = [e[-1] for e in edges if e[-1] != gap]
            if cols == []:
                colors.append('c')
            else:
                colors.append(max(set(cols), key=cols.count))
        nx.draw_networkx_nodes(G, pos, node_shape='v', node_color=[colors[i] for i in self.sensors], nodelist=self.sensors,
                               node_size=2000, alpha=1)
        nx.draw_networkx_nodes(G, pos, node_shape='o', node_color=[colors[i] for i in self.inter], nodelist=self.inter,
                               node_size=2000, alpha=1)
        nx.draw_networkx_nodes(G, pos, node_shape='H', node_color=[colors[i] for i in self.commands], nodelist=self.commands,
                               node_size=2000, alpha=1)

        nx.draw_networkx_labels(G, pos, self.labels, font_color='white', font_weight='bold')
        edges_exc = [e for e in G.edges if G[e[0]][e[1]][e[2]]['color'] == exc]
        edges_inh = [e for e in G.edges if G[e[0]][e[1]][e[2]]['color'] == inh]
        edges_gap = [e for e in G.edges if G[e[0]][e[1]][e[2]]['color'] == gap]
        style = ArrowStyle("wedge", tail_width=2., shrink_factor=0.2)
        styleg = ArrowStyle("wedge", tail_width=0.6, shrink_factor=0.4)
        nx.draw_networkx_edges(G, pos, arrowstyle=style, edgelist=edges_exc, edge_color='Chartreuse',
                               arrowsize=10, alpha=1, width=1)
        nx.draw_networkx_edges(G, pos, arrowstyle=style, edgelist=edges_inh, edge_color='red',
                               arrowsize=10, alpha=0.4, width=1)
        nx.draw_networkx_edges(G, pos, arrowstyle=styleg, edgelist=edges_gap, edge_color=gap,
                               arrowsize=10, alpha=1, width=0.1, style='dotted')
        plt.axis('off')
        utils.save_show(show, save, name='Circuit')


    def plot_output(self, *args, **kwargs):
        return self._neurons.plot_output(*args, **kwargs)

    def plots_output_mult(self, *args, **kwargs):
        labels = [self.labels[i] for i in range(len(self.labels))]
        return utils.plots_output_mult(labels=labels, *args, **kwargs)


class CircuitTf(Circuit, Optimized):
    """
    Circuit using tensorflow
    """

    def __init__(self, neurons, synapses={}, gaps={}, labels=None, sensors=set(), commands=set()):
        """

        Args:
            labels: 
            synapses(dict): initial parameters for the synapses
            neurons(NeuronModel): if not None, all other parameters except conns are ignores
        """
        Optimized.__init__(self, dt=neurons.dt)
        Circuit.__init__(self, neurons=neurons, synapses=synapses, gaps=gaps, tensors=True, labels=labels, sensors=sensors,
                         commands=commands)
        if self._num > 1:
            constraints_dic = give_constraints(synapses[0])
            #update constraint size for parallelization
            self.constraints_dic = dict(
                [(var, np.stack([val for _ in range(self._num)], axis=val.ndim)) if val.ndim > 1 else (var, val) for var, val in
                 constraints_dic.items()])
        else:
            self.constraints_dic = give_constraints(synapses)

    @classmethod
    def create_random(cls, n_neuron, syn_keys={}, gap_keys={}, n_rand=10, dt=0.1, labels=None, sensors=set(),
                      commands=set()):
        neurons = neuron.BioNeuronTf(n_rand=n_neuron, dt=dt)
        synapses = [{k: get_syn_rand(v) for k, v in syn_keys.items()} for _ in range(n_rand)]
        gaps = [{k: get_gap_rand() for k in gap_keys} for _ in range(n_rand)]
        return cls(neurons, synapses, gaps, labels, sensors, commands)


    def reset(self):
        """prepare the variables as tensors, prepare the constraints, call reset for self._neurons"""
        self._param = {}
        self.constraints = []
        for var, val in self._init_p.items():
            self._param[var] = tf.get_variable(var, initializer=val, dtype=tf.float32)
            if var in self.constraints_dic:
                # add dimension for later
                con = self.constraints_dic[var]
                self.constraints.append(tf.assign(self._param[var], tf.clip_by_value(self._param[var], con[0], con[1])))
        self._neurons.reset()
        self._init_state = self._neurons.init_state

    def build_graph(self, batch=1):
        tf.reset_default_graph()
        self.reset()
        xshape = [None]
        initializer = self._init_state.astype(np.float32)

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
            sess.run(tf.global_variables_initializer())
            results = sess.run(res_, feed_dict={
                input_cur: i
            })
        return results
    
    def settings(self):
        return ('Circuit optimization'.center(20, '.') + '\n' +
                'Chemical connections : \n %s' % (self.synapses.keys()) + '\n' +
                'Gap junctions : \n %s' % self.gaps.keys() + '\n' +
                'Initial synaptic params : %s' % self._init_p + '\n' +
                'Synaptic constraints : %s' % self.constraints_dic + '\n' +
                self._neurons.settings())

    def apply_constraints(self, session):
        session.run(self.constraints)
        self._neurons.apply_constraints(session)

    def apply_init(self, session):
        self._neurons.apply_init(session)

    @property
    def init_params(self):
        if self._neurons.trainable:
            return {**self._init_p, **self._neurons.init_params}
        else:
            return self._init_p

    @property
    def variables(self):
        if self._neurons.trainable:
            return {**self._param, **self._neurons.variables}
        else:
            return self._param

    def study_vars(self, p, show=False, save=True):
        self.plot_vars(p, func=utils.bar, suffix='compared', show=show, save=save)
        self.plot_vars(p, func=utils.box, suffix='boxes', show=show, save=save)
        if self._neurons.trainable:
            for i in range(self._neurons.num):
                self._neurons.study_vars({var: val[i] for var, val in p.items()}, suffix=self.labels[i], show=show, save=save)

    def plot_vars(self, var_dic, suffix="", show=True, save=False, func=utils.plot):
        """plot variation/comparison/boxplots of synaptic variables

        Args:
          var_dic(dict): synaptic parameters, each value of size [time, n_synapse, parallelization]
          suffix:  (Default value = "")
          show(bool): If True, show the figure (Default value = True)
          save(bool): If True, save the figure (Default value = False)
          func:  (Default value = plot)
        """

        def oneplot(var_d, name, labels):
            if func == utils.box:
                df = pd.DataFrame.from_dic(var_d)
                func(df, utils.COLORS[:len(labels)], labels)
            else:
                for i, var in enumerate(labels):
                    ax = plt.subplot(2, 2, i + 1)
                    func(ax, var_d[var])
                    plt.ylabel(var)
            plt.tight_layout()
            utils.save_show(show, save, name='{}{}_{}'.format(utils.SYN_DIR, name, suffix), dpi=300)

        if (self._num > 1):
            dim = var_dic['E'].ndim
            # if parallelization, compare run on each synapse
            for i, name in enumerate(self.synapses.keys()):
                labels = ['G', 'mdp', 'E', 'scale']
                #if data with optimization steps
                if dim > 2:
                    var_d = {l: var_dic[l][:,i] for l in labels}
                #if final result
                else:
                    var_d = {l: var_dic[l][i] for l in labels}
                oneplot(var_d, 'Synapse_{}-{}'.format(self.labels[name[0]], self.labels[name[1]]), labels)
            for i, name in enumerate(self.gaps.keys()):
                labels = ['G_gap']
                if dim > 2:
                    var_d = {l: var_dic[l][:,i] for l in labels}
                else:
                    var_d = {l: var_dic[l][i] for l in labels}
                oneplot(var_d, 'Gap_junc_{}-{}'.format(self.labels[name[0]], self.labels[name[1]]), labels)

            if self._neurons.trainable:
                for i in range(self._neurons.num):
                    if dim > 2:
                        var_d = {l: var_dic[l][:, i] for l in self._neurons.init_params.keys()}
                    else:
                        var_d = {l: var_dic[l][i] for l in self._neurons.init_params.keys()}
                    self._neurons.plot_vars(var_d, suffix='evolution_%s' % self.labels[i], show=show, save=save)

        else:
            # if not, compare all synapses together
            oneplot(var_dic, 'All_Synapses', ['G', 'mdp', 'E', 'scale'])
            oneplot(var_dic, 'All_gaps', ['G_gap'])

            if self._neurons.trainable:
                self._neurons.plot_vars(var_dic, suffix='evolution_all', show=show, save=save)

    

class CircuitFix(Circuit):

    def __init__(self, pars, dt=0.1, synapses={}, gaps={}, labels=None, sensors=set(), commands=set()):
        Circuit.__init__(self, neurons=neuron.BioNeuronFix(init_p=pars, dt=dt), synapses=synapses, gaps=gaps,
                         tensors=False, labels=labels, sensors=sensors, commands=commands)
        self._param = self._init_p

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


