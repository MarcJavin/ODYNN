"""
.. module:: Neuron
    :synopsis: Module containing classes for different neuron models

.. moduleauthor:: Marc Javin
"""


from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import tensorflow as tf

import neuron_params
import utils
from Optimizer import Optimized

V_pos = 0
Ca_pos = -1


class Model(ABC):
    """Abstract class to implement for using a new model"""
    default = None
    _constraints_dic = None
    _init_state = None

    def __init__(self, init_p=neuron_params.DEFAULT, tensors=False, dt=0.1):
        self._tensors = tensors
        if isinstance(init_p, list):
            self.num = len(init_p)
            init_p = dict([(var, np.array([p[var] for p in init_p], dtype=np.float32)) for var in init_p[0].keys()])
            self.init_state = np.stack([self.init_state for _ in range(self.num)], axis=1)
        else:
            self.num = 1
        self._param = init_p
        self.dt = dt

    @staticmethod
    @abstractmethod
    def step_model(X, i, self):
        """
        Integrate and update voltage after one time step

        Parameters
        ----------
        X : vector
            State variables
        i : float
            Input current

        Returns
        ----------
        Vector with the same size as X containing the updated state variables
        """
        pass

    @staticmethod
    @abstractmethod
    def get_random():
        """Return a dictionnary with random parameters"""
        pass

    @abstractmethod
    def calculate(self, i):
        """Iterate over i (current) and return the state variables obtained after each step"""
        pass

    def get_init_state(self):
        """Returns a vector containing the initial values of the state variables"""
        return self.init_state


class HodgkinHuxley(Model):
    """Full Hodgkin-Huxley Model implemented in Python"""

    REST_CA = neuron_params.REST_CA
    init_state = neuron_params.INIT_STATE
    default = neuron_params.DEFAULT
    _constraints_dic = neuron_params.CONSTRAINTS

    def __init__(self, init_p=neuron_params.DEFAULT, tensors=False, dt=0.1):
        Model.__init__(self, init_p=init_p, tensors=tensors, dt=dt)

    def inf(self, V, rate):
        """steady state value of a rate"""
        mdp = self._param['%s__mdp' % rate]
        scale = self._param['%s__scale' % rate]
        if self._tensors:
            # print('V : ', V)
            # print('mdp : ', mdp)
            return tf.sigmoid((V - mdp) / scale)
        else:
            return 1 / (1 + sp.exp((mdp - V) / scale))

    def h(self, cac):
        """Channel gating kinetics. Functions of membrane voltage"""
        q = self.inf(cac, 'h')
        return 1 + (q - 1) * self._param['h__alpha']

    def g_Ca(self, e, f, h):
        return self._param['g_Ca'] * e ** 2 * f * h

    def I_Ca(self, V, e, f, h):
        """
        Membrane current (in uA/cm^2)
        Calcium (Ca = element name)
        """
        return self._param['g_Ca'] * e ** 2 * f * h * (V - self._param['E_Ca'])

    def g_Kf(self, p, q):
        return self._param['g_Kf'] * p ** 4 * q

    def I_Kf(self, V, p, q):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self._param['g_Kf'] * p ** 4 * q * (V - self._param['E_K'])

    def g_Ks(self, n):
        return self._param['g_Ks'] * n

    def I_Ks(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self._param['g_Ks'] * n * (V - self._param['E_K'])

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak
        """
        return self._param['g_L'] * (V - self._param['E_L'])

    """default model"""

    @staticmethod
    def step_model(X, i_inj, self):
        """
        Integrate and update voltage after one time step
        Parameters
        ----------
        X :
        """
        V = X[V_pos]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[Ca_pos]

        h = self.h(cac)
        # V = V * (i_inj + self.g_Ca(e,f,h)*self._param['E_Ca'] + (self.g_Ks(n)+self.g_Kf(p,q))*self._param['E_K'] + self._param['g_L']*self._param['E_L']) / \
        #     ((self._param['C_m']/self.dt) + self.g_Ca(e,f,h) + self.g_Ks(n) + self.g_Kf(p,q) + self._param['g_L'])
        V += ((i_inj - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self._param[
            'C_m']) * self.dt

        cac += (-self.I_Ca(V, e, f, h) * self._param['rho_ca'] - (
                    (cac - self.REST_CA) / self._param['decay_ca'])) * self.dt
        tau = self._param['p__tau']
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self.inf(V, 'p') / tau))
        tau = self._param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self.inf(V, 'q') / tau))
        tau = self._param['e__tau']
        e = ((tau * self.dt) / (tau + self.dt)) * ((e / self.dt) + (self.inf(V, 'e') / tau))
        tau = self._param['f__tau']
        f = ((tau * self.dt) / (tau + self.dt)) * ((f / self.dt) + (self.inf(V, 'f') / tau))
        tau = self._param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self.inf(V, 'n') / tau))

        if self._tensors:
            return tf.stack([V, p, q, n, e, f, cac], 0)
        else:
            return [V, p, q, n, e, f, cac]

    def get_random(self):
        return neuron_params.give_rand()

    @staticmethod
    def plot_vars(*args, **kwargs):
        return utils.plot_vars(*args, **kwargs)

    @staticmethod
    def no_tau_ca(X, i_inj, self):
        """
        Integrate
        """
        V = X[V_pos]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[Ca_pos]
        h = self.h(cac)
        V += ((i_inj - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(
            V)) / self._param['C_m']) * self.dt
        cac += (-self.I_Ca(V, e, f, h) * self._param['rho_ca'] - (
                (cac - self.REST_CA) / self._param['decay_ca'])) * self.dt
        tau = self._param['p__tau']
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self.inf(V, 'p') / tau))
        tau = self._param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self.inf(V, 'q') / tau))
        tau = self._param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self.inf(V, 'n') / tau))
        e = self.inf(V, 'e')
        f = self.inf(V, 'f')
        if self._tensors:
            return tf.stack([V, p, q, n, e, f, cac], 0)
        else:
            return [V, p, q, n, e, f, cac]

    @staticmethod
    def no_tau(X, i_inj, self):
        """
        Integrate
        """
        V = X[V_pos]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[Ca_pos]
        h = self.h(cac)
        V += ((i_inj - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(
            V)) / self._param['C_m']) * self.dt
        # cac = (self._param['decay_ca'] / (self.dt + self._param['decay_ca'])) * (
        #             cac - self.I_Ca(V, e, f, h) * self._param['rho_ca'] * self.dt + self.REST_CA * self._param['decay_ca'] / self.dt)
        # 
        # cac += (-self.I_Ca(V, e, f, h) * self._param['rho_ca'] - (
        #             (cac - self.REST_CA) / self._param['decay_ca'])) * self.dt
        p = self.inf(V, 'p')
        q = self.inf(V, 'q')
        e = self.inf(V, 'e')
        f = self.inf(V, 'f')
        n = self.inf(V, 'n')
        if self._tensors:
            return tf.stack([V, p, q, n, e, f, cac], 0)
        else:
            return [V, p, q, n, e, f, cac]

    @staticmethod
    def ica_from_v(X, v_fix, self):
        e = X[1]
        f = X[2]
        cac = X[Ca_pos]

        h = self.h(cac)
        tau = self._param['e__tau']
        e = ((tau * self.dt) / (tau + self.dt)) * ((e / self.dt) + (self.inf(v_fix, 'e') / tau))
        tau = self._param['f__tau']
        f = ((tau * self.dt) / (tau + self.dt)) * ((f / self.dt) + (self.inf(v_fix, 'f') / tau))
        ica = self.I_Ca(v_fix, e, f, h)
        cac += (-self.I_Ca(v_fix, e, f, h) * self._param['rho_ca'] - (
                (cac - self.REST_CA) / self._param['decay_ca'])) * self.dt

        if self._tensors:
            return tf.stack([ica, e, f, h, cac], 0)
        else:
            return [ica, e, f, h, cac]

    @staticmethod
    def ik_from_v(X, v_fix, self):
        p = X[1]
        q = X[2]
        n = X[3]

        tau = self._param['p__tau']
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self.inf(v_fix, 'p') / tau))
        tau = self._param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self.inf(v_fix, 'q') / tau))
        tau = self._param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self.inf(v_fix, 'n') / tau))
        ik = self.I_Kf(v_fix, p, q) + self.I_Ks(v_fix, n)

        if self._tensors:
            return tf.stack([ik, p, q, n], 0)
        else:
            return [ik, p, q, n]


MODEL = HodgkinHuxley


class NeuronTf(MODEL, Optimized):
    nb = -1

    def __init__(self, init_p=neuron_params.DEFAULT, dt=0.1, fixed=[], constraints=neuron_params.CONSTRAINTS):
        HodgkinHuxley.__init__(self, init_p=init_p, tensors=True, dt=dt)
        Optimized.__init__(self)
        self.init_p = self._param
        self._fixed = fixed
        self._constraints_dic = constraints
        self.id = self.give_id()

    @classmethod
    def give_id(cls):
        cls.nb += 1
        return str(cls.nb)

    def step(self, hprev, x):
        return self.step_model(hprev, x, self)

    def _reset(self):
        """rebuild tf variable graph"""
        with(tf.variable_scope(self.id)):
            self._param = {}
            self._constraints = []
            for var, val in self.init_p.items():
                if var in self._fixed:
                    self._param[var] = tf.constant(val, name=var, dtype=tf.float32)
                else:
                    self._param[var] = tf.get_variable(var, initializer=val, dtype=tf.float32)
                    if var in self._constraints_dic:
                        con = self._constraints_dic[var]
                        self._constraints.append(
                            tf.assign(self._param[var], tf.clip_by_value(self._param[var], con[0], con[1])))
        # print('neuron_params after reset : ', self._param)

    def parallelize(self, n):
        """Add a dimension of size n in the parameters"""
        if self.num > 1:
            self.init_p = dict(
                [(var, np.stack([val for _ in range(n)], axis=val.ndim)) for var, val in self.init_p.items()])
        else:
            self.init_p = dict(
                [(var, np.stack([val for _ in range(n)], axis=0)) for var, val in self.init_p.items()])
        self.init_state = np.stack([self.init_state for _ in range(n)], axis=self.init_state.ndim)

    def build_graph(self, batch=None):
        tf.reset_default_graph()
        self._reset()
        xshape = [None]
        initializer = self.init_state
        if batch is not None:
            xshape.append(None)
            initializer = np.stack([initializer for _ in range(batch)], axis=1)
        if self.num > 1:
            xshape.append(self.num)
        curs_ = tf.placeholder(shape=xshape, dtype=tf.float32, name='input_current')
        res_ = tf.scan(self.step,
                       curs_,
                       initializer=initializer.astype(np.float32))
        return curs_, res_

    def calculate(self, i):
        if i.ndim > 1:
            input_cur, res_ = self.build_graph(batch=i.shape[1])
        else:
            input_cur, res_ = self.build_graph()
        if i.ndim < 3 and self.num > 1:
            i = np.stack([i for _ in range(self.num)], axis=i.ndim)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run(res_, feed_dict={
                input_cur: i
            })
        return results

    def settings(self):
        return ('Neuron optimization'.center(20, '.') + '\n' +
                'Nb of neurons : {}'.format(self.num) + '\n' +
                'Initial neuron params : {}'.format(self.init_p) + '\n' +
                'Fixed variables : {}'.format([c for c in self._fixed]) + '\n' +
                'Initial state : {}'.format(self.init_state) + '\n' +
                'Constraints : {}'.format(self._constraints_dic) + '\n' +
                'dt : {}'.format(self.dt) + '\n')

    def apply_constraints(self, session):
        session.run(self._constraints)

    def get_params(self):
        return self._param.items()


class NeuronLSTM(Optimized):
    num = 1
    _cell_size = 2
    _hidden_layer_nb = 2
    _hidden_layer_cells = 5
    _hidden_layer_size = 10
    init_p = {}

    _max_cur = 60
    _min_v = -60
    _scale_v = 100
    _scale_ca = 1000

    def __init__(self, dt=0.1):
        Optimized.__init__(self)
        self.dt = dt

    def build_graph(self, batch=None):
        tf.reset_default_graph()
        xshape = [None, None]
        if batch is None:
            batch = 1

        curs_ = tf.placeholder(shape=xshape, dtype=tf.float32, name='input_current')
        input = tf.expand_dims(curs_ / self._max_cur, axis=len(xshape))

        for layer in range(self._hidden_layer_nb):
            hidden = []
            for cell in range(self._hidden_layer_cells):
                out, st = self._lstm_cell(self._hidden_layer_size, input, batch, '{}-{}'.format(layer, cell))
                hidden.append(out)
            hidden = tf.reduce_sum(tf.stack(hidden, axis=0), axis=0)
            input = hidden

        rnn_outputs, rnn_states = self._lstm_cell(self._cell_size, input, batch, 'post_V_Ca')
        with tf.name_scope('Scale'):
            res_ = tf.transpose(rnn_outputs, perm=[0, 2, 1])
            V = res_[:, V_pos] * self._scale_v + self._min_v
            Ca = res_[:, Ca_pos] * self._scale_ca
            results = tf.stack([V, Ca], axis=1)

        return curs_, results

    def calculate(self, i):
        pass

    @staticmethod
    def _lstm_cell(size, input, batch, scope):
        with tf.variable_scope(scope):
            cell = tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
            initializer = cell.zero_state(batch, dtype=tf.float32)
            return tf.nn.dynamic_rnn(cell, inputs=input, initial_state=initializer, time_major=True)

    def settings(self):
        return ('Cell size : {} '.format(self._cell_size) + '\n' +
                'Number of hidden layers : {}'.format(self._hidden_layer_nb) + '\n'
                'Number of hidden cells : {}'.format(self._hidden_layer_cells) + '\n' +
                'State size in hidden layer : {}'.format(self._hidden_layer_size) + '\n' +
                'dt : {}'.format(self.dt) + '\n' +
                'max current : {}, min voltage : {}, scale voltage : {}, scale calcium : {}'
                .format(self._max_cur, self._min_v, self._scale_v, self._scale_ca)
                )


class NeuronFix(MODEL):

    def __init__(self, init_p=neuron_params.DEFAULT, dt=0.1):
        HodgkinHuxley.__init__(self, init_p=init_p, tensors=False, dt=dt)
        self.state = self.init_state

    def init_batch(self, n):
        self.init_state = np.stack([self.init_state for _ in range(n)], axis=1)

    def step(self, i):
        self.state = np.array(self.step_model(self.state, i, self))
        return self.state

    def calculate(self, i_inj, currents=False):
        X = []
        self.reset()
        for i in i_inj:
            X.append(self.step(i))
        return np.array(X)

    def reset(self):
        self.state = self.init_state
