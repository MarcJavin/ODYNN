"""
.. module:: optimize
    :synopsis: Module containing classes for optimization with Tensorflow

.. moduleauthor:: Marc Javin
"""

import pickle
import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import copy
from tqdm import tqdm

from .utils import OUT_SETTINGS, IMG_DIR
from . import utils
import pylab as plt

SAVE_PATH = utils.TMP_DIR + 'model.ckpt'
FILE_LV = utils.TMP_DIR + 'dump_lossratevars'
FILE_OBJ = utils.TMP_DIR + 'optimized'
TRAIN_FILE = utils.TMP_DIR + 'train'
TEST_FILE = utils.TMP_DIR + 'test'

INTRA_PAR = 4
INTER_PAR = 1


class Optimized(ABC):
    """Abstract class for object to be optimized. It could represent on or a set of neurons, or a circuit."""

    ions = {}
    _num = None

    def __init__(self, dt):
        self.dt = dt
        self._init_p = {}
        self._param = {}
        self._constraints = {}

    @property
    def num(self):
        return self._num

    @abstractmethod
    def build_graph(self, batch=1):
        """Build the tensorflow graph. Take care of the loop and the initial state."""
        pass

    @abstractmethod
    def settings(self):
        """
        Give a string describing the settings
        Returns(str): description

        """
        return ''

    @staticmethod
    def plot_vars(var_dic, suffix, show, save):
        """A function to plot the variables of the optimized object

        Args:
          var_dic: 
          suffix: 
          show: 
          save:

        """
        pass

    def apply_constraints(self, session):
        """
        Apply necessary constraints to the optimized variables

        Args:
          session(tf.Session):

        """
        pass

    def apply_init(self, session):
        pass

    @property
    def init_params(self):
        return self._init_p

    @init_params.setter
    def init_params(self, value):
        self._init_p = value

    @property
    def variables(self):
        return {}

    def predump(self, sess):
        pass

    def study_vars(self, p, *args, **kwargs):
        pass


class Optimizer(ABC):

    def __init__(self, optimized, frequency=30):
        """

        Args:
            optimized(Optimized):
            epochs:
            frequency:
        """
        self.start_time = time.time()
        self.optimized = optimized
        self._parallel = self.optimized.num
        self.dir = None
        self._loss = None
        self.freq_test = 30
        self._test_losses = None
        self._test = False

    def _init_l_rate(self):
        global_step = tf.Variable(0, trainable=False)
        # progressive learning rate
        self.learning_rate = tf.train.exponential_decay(
            self.l_rate[0],  # Base learning rate.
            global_step,  # Current index to the dataset.
            self.l_rate[1],  # Decay step.
            self.l_rate[2],  # Decay rate.
            staircase=True)
        tf.summary.scalar("learning rate", self.learning_rate)
        return global_step

    def _build_train(self):
        """learning rate and optimization"""
        global_step = self._init_l_rate()
        # self.learning_rate = 0.1
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gvs = opt.compute_gradients(self._loss)
        gvs = [(g,v) for g,v in gvs if g is not None]
        grads, vars = zip(*gvs)

        if self._parallel > 1:
            grads_normed = []
            for i in range(self._parallel):
                # clip by norm for each parallel model (neuron or circuit)
                gi = [g[..., i] for g in grads]
                # gi = [tf.where(tf.is_nan(g), tf.zeros_like(g), g) for g in gi]
                gi_norm, _ = tf.clip_by_global_norm(gi, 5.)
                grads_normed.append(gi_norm)
            grads_normed = [tf.stack([grads_normed[i][j] for i in range(self._parallel)], axis=-1) for j in range(len(grads))]
        else:
            grads_normed, _ = tf.clip_by_global_norm(grads, 5.)
        self.train_op = opt.apply_gradients(zip(grads_normed, vars), global_step=global_step)
        # tf.summary.histogram(name='gradients', values=gvs)

        self.saver = tf.train.Saver()

    def _init(self, dir, suffix, train, test, l_rate, w, yshape):
        """Initialize directory and the object to be optimized, get the dataset, write settings in the directory
        and initialize placeholders for target output and results.

        Args:
          dir(str): path to the directory          suffix:
          train: 
          test: 
          l_rate: 
          w: 
          yshape:

        Raises:
            ValueError: if the timestep of the optimized object is different than the one in train or test

        """
        self.dir = dir
        self.suffix = suffix
        tf.reset_default_graph()
        self.l_rate = l_rate

        if test is not None:
            self._test = True
            self._test_losses = []
            if self.optimized.dt != test[0][1] - test[0][0]:
                raise ValueError('The expected time step from the optimized object is {}, but got {} in the test data'
                                 .format(self.optimized.dt, test[0][1] - test[0][0]))
        if self.optimized.dt != train[0][1] - train[0][0]:
            raise ValueError(
                'The expected time step from the optimized object is {}, but got {} in the train data'.format(
                    self.optimized.dt, train[0][1] - train[0][0]))

        self.n_batch = train[1].shape[1]
        sett = self.settings(w, train)
        with open(self.dir + OUT_SETTINGS, 'w') as f:
            f.write(sett)
        with open(self.dir + TRAIN_FILE, 'wb') as f:
            pickle.dump(train, f)
        with open(self.dir + TEST_FILE, 'wb') as f:
            pickle.dump(test, f)

        if self._parallel > 1:
            # add dimension to current for neurons trained in parallel
            train[1] = np.stack([train[1] for _ in range(self._parallel)], axis=-1)

            if self._test:
                test[1] = np.stack([test[1] for _ in range(self._parallel)], axis=-1)

        xs_, res = self.optimized.build_graph(batch=self.n_batch)
        ys_ = [tf.placeholder(shape=yshape, dtype=tf.float32, name="Measure_out_%s"%i) if t is not None
                    else 0. for i,t in enumerate(train[-1])]

        print("i expected : ", xs_.shape)
        print("i : ", train[1].shape)

        return xs_, ys_, res, train, test

    def settings(self, w, train):
        """Give the settings of the optimization

        Args:
          w(tuple): weights for the loss of voltage and ions concentrations

        Returns:
            str: settings
        """
        show_shape = None
        for t in train[-1]:
            if t is not None:
                show_shape = t.shape
                break

        return ("Weights (out, cac) : {}".format(w) + "\n" +
                "Start rate : {}, decay_step : {}, decay_rate : {}".format(self.l_rate[0], self.l_rate[1],
                                                                           self.l_rate[2]) + "\n" +
                "Number of batches : {}".format(self.n_batch) + "\n" +
                "Number of time steps : {}".format(train[0].shape) + "Input current shape : {}".format(
                train[1].shape) +
                "Output shape : {}".format(show_shape) + "\n" +
                "Number of models : {}".format(self._parallel) + '\n' +
                self.optimized.settings())

    def plot_out(self, *args, **kwargs):
        pass

    @abstractmethod
    def _build_loss(self, res, ys_, w):
        pass

    def optimize(self, dir, train_=None, test_=None, w=None, epochs=700, l_rate=(0.1, 9, 0.92), suffix='', step='',
                 reload=False, reload_dir=None, yshape=None, evol_var=True, plot=True):

        print('Optimization'.center(40,'_'))

        T, X, res_targ = train_
        if w is None:
            w = [1] + [0 for _ in range(1, len(res_targ))]
        if (len(w) != len(res_targ)):
            raise ValueError('The number of measurable variables and weights must be the same')
        w = [wi if res_targ[i] is not None else 0 for i, wi in enumerate(w)]

        if test_ is not None:
            T_test, X_test, res_targ_test = test_

        if reload_dir is None:
            reload_dir = dir

        xs_, ys_, res, train, test = self._init(dir, suffix, copy.deepcopy(train_), copy.deepcopy(test_), l_rate, w, yshape)

        self._build_loss(res, ys_, w)
        self._build_train()
        self.summary = tf.summary.merge_all()
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=INTRA_PAR,
            inter_op_parallelism_threads=INTER_PAR)

        with tf.Session(config=session_conf) as sess:

            self.tdb = tf.summary.FileWriter(self.dir + '/tensorboard',
                                             sess.graph)

            sess.run(tf.global_variables_initializer())
            losses = np.zeros((epochs, self._parallel))
            rates = np.zeros(epochs)

            if reload:
                """Get variables and measurements from previous steps"""
                self.saver.restore(sess, '%s/%s' % (reload_dir, SAVE_PATH + suffix))
                with open(reload_dir + '/' + FILE_LV + suffix, 'rb') as f:
                    l, self._test_losses, r, vars = pickle.load(f)
                losses = np.concatenate((l, losses))
                rates = np.concatenate((r, rates))
                # sess.run(tf.assign(self.global_step, 200))
                len_prev = len(l)
            else:
                vars = {var : np.array([val]) for var, val in self.optimized.init_params.items()}
                len_prev = 0

            if(evol_var):
                vars = {var: np.vstack((val, np.zeros([epochs] + list(val.shape)[1:]))) for var, val in vars.items()}
            else:
                vars = {var: np.vstack((val, np.zeros([1] + list(val.shape)[1:]))) for var, val in vars.items()}

            for j in tqdm(range(epochs)):
                i = len_prev + j
                feed_d = {ys_[i]: m for i,m in enumerate(train[-1]) if m is not None}
                feed_d[xs_] = train[1]
                summ, results, _, train_loss = sess.run([self.summary, res, self.train_op, self._loss], feed_dict=feed_d)

                self.tdb.add_summary(summ, i)

                self.optimized.apply_constraints(sess)

                if evol_var or j == epochs - 1:
                    for name, v in self.optimized.variables.items():
                        v_ = sess.run(v)
                        if evol_var:
                            vars[name][i + 1] = v_
                        else:
                            vars[name][-1] = v_

                rates[i] = sess.run(self.learning_rate)
                losses[i] = train_loss
                if self._parallel > 1:
                    train_loss = np.nanmean(train_loss)
                print("[{}] loss : {}".format(i, train_loss))

                if plot:
                    self.plot_out(X, results, res_targ, suffix, step, 'train', i)

                if i % self.freq_test == 0 or j == epochs - 1:
                    res_test = None
                    if test is not None:
                        feed_d = {ys_[i]: m for i,m in enumerate(test[-1]) if m is not None}
                        feed_d[xs_] = test[1]
                        test_loss, res_test = sess.run([self._loss, res], feed_dict=feed_d)
                        self._test_losses.append(test_loss)

                    with (open(self.dir + FILE_LV + self.suffix, 'wb')) as f:
                        pickle.dump([losses, self._test_losses, rates, vars], f)

                    self.saver.save(sess, "{}{}{}".format(self.dir, SAVE_PATH, self.suffix))

                    if plot:
                        plot_loss_rate(losses[:i + 1], rates[:i + 1], losses_test=self._test_losses,
                                       parallel=self._parallel, suffix=self.suffix, show=False, save=True)
                        if evol_var:
                            self.optimized.plot_vars(dict([(name, val[:i + 2]) for name, val in vars.items()]),
                                                 suffix=self.suffix + "evolution", show=False, save=True)
                    if res_test is not None and plot:
                        self.plot_out(X, res_test, res_targ_test, suffix, step, 'test', i)

                    with open(self.dir + FILE_OBJ, 'wb') as f:
                        self.optimized.predump(sess)
                        pickle.dump(self.optimized, f)


            with open(self.dir + 'time', 'w') as f:
                f.write(str(time.time() - self.start_time))

        # plot evolution of variables
        p = get_vars(self.dir)
        self.optimized.study_vars(p)
        return self.optimized

def get_train(dir):
    file = dir + '/' + TRAIN_FILE
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_model(dir):
    file = dir + '/' + FILE_OBJ
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_vars(dir, i=-1, loss=False):
    """get dic of vars from dumped file

    Args:
      dir(str): path to the directory      i:  (Default value = -1)

    Returns:

    """
    file = dir + '/' + FILE_LV
    with open(file, 'rb') as f:
        load = pickle.load(f, encoding="latin1")
        l = load[0]
        dic = load[-1]
        if loss:
            dic['loss'] = l
        dic = dict([(var, np.array(val[i], dtype=np.float32)) for var, val in dic.items()])
    return dic


def get_vars_all(dir, i=-1):
    """get dic of vars from dumped file

    Args:
      dir(str): path to the directory      i:  (Default value = -1)

    Returns:

    """
    file = dir + '/' + FILE_LV
    with open(file, 'rb') as f:
        dic = pickle.load(f)[-1]
        dic = dict([(var, val[:i]) for var, val in dic.items()])
    return dic


def get_best_result(dir, i=-1, loss=False):
    """

    Args:
      dir(str): path to the directory      i:  (Default value = -1)

    Returns:
      

    """
    file = dir + '/' + FILE_LV
    with open(file, 'rb') as f:
        load = pickle.load(f, encoding="latin1")
        l = load[0]
        dic = load[-1]
        idx = np.nanargmin(l[-1])
        # [epoch, model] for neuron, [epoch, element, model] for circuit
        ndim = list(dic.values())[0].ndim
        if l.shape[1] > 1:
            if ndim > 2:
                dic = dict([(var, val[i, :, idx]) for var, val in dic.items()])
            else:
                dic = dict([(var, val[i,idx]) for var, val in dic.items()])
        else:
            dic = dict([(var, val[i]) for var, val in dic.items()])
        if (loss):
            dic['loss'] = l[i, idx]
    return dic


def plot_loss_rate(losses, rates, losses_test=None, parallel=1, suffix="", show=False, save=True):
    """plot loss (log10) and learning rate

    Args:
      losses: 
      rates: 
      losses_test:  (Default value = None)
      parallel:  (Default value = 1)
      suffix:  (Default value = "")
      show(bool): If True, show the figure (Default value = False)
      save:  (Default value = True)

    Returns:

    """
    plt.figure()

    n_p = 2
    if losses_test is not None and parallel > 1:
        n_p = 3

    plt.ylabel('Test Loss')
    plt.yscale('log')

    plt.subplot(n_p,1,1)
    if parallel == 1:
        plt.plot(losses, 'r', linewidth=0.6, label='Train')
    else:
        plt.plot(losses, linewidth=0.6)
    plt.ylabel('Loss')
    if losses_test is not None:
        losses_test = np.array(losses_test)
        pts = np.linspace(0, losses.shape[0]-1, num=losses_test.shape[0], endpoint=True)
        if parallel == 1:
            plt.plot(pts, losses_test, 'Navy', linewidth=0.6, label='Test')
            plt.legend()
        else:
            # add another subplot for readability
            n_p = 3
            plt.ylabel('Train Loss')
            plt.yscale('log')
            plt.subplot(n_p,1,2)
            plt.plot(pts, losses_test, linewidth=0.6)
            plt.ylabel('Test Loss')
    plt.yscale('log')


    plt.subplot(n_p,1,n_p)
    plt.plot(rates)
    plt.ylabel('Learning rate')

    utils.save_show(show, save, name='Losses_{}'.format(suffix), dpi=300)
