"""
.. module:: Neuron
    :synopsis: Module containing classes for neuron models

.. moduleauthor:: Marc Javin
"""


import numpy as np
import tensorflow as tf

from . import config
from .model import V_pos, Ca_pos
from .optimize import Optimized


MODEL = config.NEURON_MODEL


class NeuronTf(MODEL, Optimized):
    """
    Class representing a neuron, implemented using Tensorflow.
    This class allows simulation and optimization
    """
    nb = -1

    def __init__(self, init_p=None, dt=0.1, fixed=[], constraints=None):
        MODEL.__init__(self, init_p=init_p, tensors=True, dt=dt)
        Optimized.__init__(self)
        self.init_p = self._param
        self._fixed = fixed
        if(fixed == 'all'):
            self._fixed = set(self.init_p.keys())
        if(constraints is not None):
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
        self._init_state = np.stack([self._init_state for _ in range(n)], axis=self._init_state.ndim)

    def build_graph(self, batch=None):
        tf.reset_default_graph()
        self._reset()
        xshape = [None]
        initializer = self._init_state
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
                'Initial state : {}'.format(self._init_state) + '\n' +
                'Constraints : {}'.format(self._constraints_dic) + '\n' +
                'dt : {}'.format(self.dt) + '\n')

    def apply_constraints(self, session):
        session.run(self._constraints)

    def get_params(self):
        return self._param.items()


class NeuronLSTM(Optimized):
    """
    Behavior model of a neuron using an LSTM network
    """
    num = 1
    _cell_size = 2
    init_p = {}

    _max_cur = 60.
    _min_v = -60.
    _scale_v = 100.
    _scale_ca = 1000.

    def __init__(self, dt=0.1, nb_layer=3, nb_cells=5, layer_size=10, extra_ca=True):
        self._hidden_layer_nb = nb_layer
        self._hidden_layer_cells = nb_cells
        self._hidden_layer_size = layer_size
        self._extra_ca = extra_ca
        Optimized.__init__(self)
        self.dt = dt

    def build_graph(self, batch=None):
        tf.reset_default_graph()
        xshape = [None, None]
        if batch is None:
            batch = 1

        curs_ = tf.placeholder(shape=xshape, dtype=tf.float32, name='input_current')
        with tf.variable_scope('prelayer'):
            input = tf.expand_dims(curs_ / self._max_cur, axis=len(xshape))

        for layer in range(self._hidden_layer_nb):
            hidden = []
            for cell in range(self._hidden_layer_cells):
                out, st = self._lstm_cell(self._hidden_layer_size, input, batch, '{}-{}'.format(layer, cell))
                hidden.append(out)
            with tf.variable_scope('interlayer{}'.format(layer)):
                hidden = tf.reduce_sum(tf.stack(hidden, axis=0), axis=0)
                input = hidden

        v_outputs, v_states = self._lstm_cell(1, input, batch, 'post_V')

        if(self._extra_ca):
            hidden = []
            for cell in range(self._hidden_layer_cells):
                out, st = self._lstm_cell(self._hidden_layer_size, v_outputs, batch, '{}-{}'.format(self._hidden_layer_nb+1, cell))
                hidden.append(out)
            with tf.variable_scope('interlayer{}'.format(self._hidden_layer_nb+1)):
                hidden = tf.reduce_sum(tf.stack(hidden, axis=0), axis=0)
                input = hidden

            ca_outputs, ca_states = self._lstm_cell(1, input, batch, 'post_Ca')

        else:

            ca_outputs, ca_states = self._lstm_cell(1, v_outputs, batch, 'post_Ca')

        with tf.name_scope('Scale'):
            V = v_outputs[:, :, V_pos] * self._scale_v + self._min_v
            Ca = ca_outputs[:, :, Ca_pos] * self._scale_ca
            results = tf.stack([V, Ca], axis=1)

        return curs_, results

    def calculate(self, i):
        pass

    @staticmethod
    def _lstm_cell(size, input, batch, scope):
        with tf.variable_scope(scope):
            cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=True, state_is_tuple=True)
            initializer = cell.zero_state(batch, dtype=tf.float32)
            return tf.nn.dynamic_rnn(cell, inputs=input, initial_state=initializer, time_major=True)

    def settings(self):
        return ('Cell size : {} '.format(self._cell_size) + '\n' +
                'Number of hidden layers : {}'.format(self._hidden_layer_nb) + '\n'
                'Number of hidden cells per layer : {}'.format(self._hidden_layer_cells) + '\n' +
                'Extra layer for [Ca] : %s' % self._extra_ca + '\n' +
                'State size in hidden layer : {}'.format(self._hidden_layer_size) + '\n' +
                'dt : {}'.format(self.dt) + '\n' +
                'max current : {}, min voltage : {}, scale voltage : {}, scale calcium : {}'
                .format(self._max_cur, self._min_v, self._scale_v, self._scale_ca)
                )


class NeuronFix(MODEL):
    """
    Class representing a neuron, implemented only in Python
    This class allows simulation but not optimization
    """

    def __init__(self, init_p=MODEL.default_params, dt=0.1):
        MODEL.__init__(self, init_p=init_p, tensors=False, dt=dt)
        self.state = self._init_state

    def init_batch(self, n):
        self._init_state = np.stack([self._init_state for _ in range(n)], axis=1)

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
        self.state = self._init_state
