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


class Optimized(ABC):
    """Abstract class for object to be optimized. It could represent on or a set of neurons, or a circuit."""

    ions = {}
    _num = None

    def __init__(self, dt):
        self.dt = dt

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

    def reset(self):
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
        return {}

    @property
    def variables(self):
        return {}

    def todump(self, sess):
        return []

    def study_vars(self, p):
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
        self.frequency = frequency
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
        return global_step

    def _build_train(self, global_step):
        """learning rate and optimization"""
        self._init_l_rate()
        # self.learning_rate = 0.1
        tf.summary.scalar("learning rate", self.learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gvs = opt.compute_gradients(self._loss)
        grads, vars = zip(*gvs)

        if self._parallel > 1:
            grads_normed = []
            for i in range(self._parallel):
                # clip by norm for each parallel model (neuron or circuit)
                gi = [g[..., i] for g in grads]
                gi_norm, _ = tf.clip_by_global_norm(gi, 5.)
                grads_normed.append(gi_norm)
            grads_normed = [tf.stack([grads_normed[i][j] for i in range(self._parallel)], axis=-1) for j in range(len(grads))]
        else:
            grads_normed, _ = tf.clip_by_global_norm(grads, 5.)
        self.train_op = opt.apply_gradients(zip(grads_normed, vars), global_step=global_step)

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

        if self._parallel > 1:
            # add dimension for neurons trained in parallel
            # [time, n_batch, neuron]
            for i in range(1, len(train)):
                train[i] = np.stack([train[i] for _ in range(self._parallel)], axis=train[i].ndim)

            if self._test:
                for i in range(1, len(test)):
                    test[i] = np.stack([test[i] for _ in range(self._parallel)], axis=test[i].ndim)
            yshape.append(self._parallel)

        self.xs_, self.res = self.optimized.build_graph(batch=self.n_batch)
        self.ys_ = tf.placeholder(shape=yshape, dtype=tf.float32, name="voltage_Ca")

        print("i expected : ", self.xs_.shape)
        print("i : ", train[1].shape)

        return train, test

    def settings(self, w, train):
        """Give the settings of the optimization

        Args:
          w(tuple): weights for the loss of voltage and ions concentrations

        Returns:
            str: settings
        """
        show_shape = train[2]
        if train[2] is None:
            show_shape = train[-1]
        return ("Weights (out, cac) : {}".format(w) + "\n" +
                "Start rate : {}, decay_step : {}, decay_rate : {}".format(self.l_rate[0], self.l_rate[1],
                                                                           self.l_rate[2]) + "\n" +
                "Number of batches : {}".format(self.n_batch) + "\n" +
                "Number of time steps : {}".format(train[0].shape) + "Input current shape : {}".format(
                train[1].shape) +
                "Output shape : {}".format(show_shape.shape) + "\n" +
                "Number of models : {}".format(self._parallel) + '\n' +
                self.optimized.settings())



    def _plots_dump(self, sess, test, losses, rates, vars, i, plot):
        """Plot the variables evolution, the loss and saves it in a file

        Args:
          sess(tf.Session): tensorflow session
          losses(ndarray): array of losses
          rates(ndarray): array containing the registered values of the learning rate
          vars(dict): dictionary with the registered value of optimized variables
          i: step in the optimization

        Returns:
            array: results of the test

        """
        results = None
        if test is not None:
            test_loss, results = sess.run([self._loss, self.res], feed_dict={
                self.xs_: test[1],
                self.ys_: np.array([test[2], test[-1]])
            })
            self._test_losses.append(test_loss)


        with (open(self.dir + FILE_LV + self.suffix, 'wb')) as f:
            pickle.dump([losses, self._test_losses, rates, vars], f)
        with open(self.dir + FILE_OBJ + self.suffix, 'wb') as f:
            pickle.dump(self.optimized.todump(sess), f)

        self.saver.save(sess, "{}{}{}".format(self.dir, SAVE_PATH, self.suffix))

        if plot:
            plot_loss_rate(losses[:i + 1], rates[:i + 1], losses_test=self._test_losses, parallel=self._parallel, suffix=self.suffix, show=False,
                           save=True)
            self.optimized.plot_vars(dict([(name, val[:i + 2]) for name, val in vars.items()]),
                                 suffix=self.suffix + "evolution", show=False,
                                 save=True)
        return results

    def plot_out(self, *args, **kwargs):
        pass

    @abstractmethod
    def _build_loss(self, w):
        pass

    def optimize(self, dir, train_=None, test_=None, w=(1, 0), epochs=700, l_rate=(0.1, 9, 0.92), suffix='', step='',
                 reload=False, reload_dir=None, yshape=None, plot=True):

        print('Optimization'.center(40,'_'))
        import psutil
        p = psutil.Process()
        # print('%s MB 1 '%(p.memory_info().vms>>20))
        T, X, V, Ca = train_
        res_targ = [V, Ca]
        if test_ is not None:
            T_test, X_test, V_test, Ca_test = test_
            res_targ_test = [V_test, Ca_test]

        if reload_dir is None:
            reload_dir = dir
        train, test = self._init(dir, suffix, copy.copy(train_), copy.copy(test_), l_rate, w, yshape)
        # print('%s MB parall'%(p.memory_info().vms>>20))
        # print(self.settings(w))

        global_step = self._build_loss(w)
        self._build_train(global_step)
        self.summary = tf.summary.merge_all()

        with tf.Session() as sess:

            Vt = train[2] if train[2] is not None else np.zeros(train[-1].shape)

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

            print('%s MB before vars'%(p.memory_info().vms>>20))
            vars = {var: np.vstack((val, np.zeros([epochs] + list(val.shape)[1:]))) for var, val in vars.items()}
            print('%s MB after vars'%(p.memory_info().vms>>20))
            for j in tqdm(range(epochs)):
                i = len_prev + j
                summ, results, _, train_loss = sess.run([self.summary, self.res, self.train_op, self._loss], feed_dict={
                    self.xs_: train[1],
                    self.ys_: np.array([Vt, train[-1]])
                })
                print('%s MB after tf'%(p.memory_info().vms>>20))

                self.tdb.add_summary(summ, i)

                self.optimized.apply_constraints(sess)

                # with open("{}{}_{}.txt".format(self.dir, OUT_PARAMS, self.suffix), 'w') as f:
                for name, v in self.optimized.variables.items():
                    v_ = sess.run(v)
                    # f.write("{} : {}\n".format(name, v_))
                    vars[name][i + 1] = v_

                rates[i] = sess.run(self.learning_rate)
                losses[i] = train_loss
                if self._parallel > 1:
                    train_loss = np.nanmean(train_loss)
                print("[{}] loss : {}".format(i, train_loss))

                if plot:
                    self.plot_out(X, results, res_targ, suffix, step, 'train', i)

                if i % self.frequency == 0 or j == epochs - 1:
                    res_test = self._plots_dump(sess, test, losses, rates, vars, len_prev + i, plot)
                    if res_test is not None and plot:
                        self.plot_out(X, res_test, res_targ_test, suffix, step, 'test', i)

            with open(self.dir + 'time', 'w') as f:
                f.write(str(time.time() - self.start_time))

        # plot evolution of variables
        p = get_vars(self.dir)
        if plot:
            self.optimized.study_vars(p)
        return self.optimized




def get_vars(dir, i=-1):
    """get dic of vars from dumped file

    Args:
      dir(str): path to the directory      i:  (Default value = -1)

    Returns:

    """
    file = dir + '/' + FILE_LV
    with open(file, 'rb') as f:
        dic = pickle.load(f)[-1]
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


def get_best_result(dir, i=-1):
    """

    Args:
      dir(str): path to the directory      i:  (Default value = -1)

    Returns:
      

    """
    file = dir + '/' + FILE_LV
    with open(file, 'rb') as f:
        l = pickle.load(f)[0]
        dic = pickle.load(f)[-1]
        idx = np.nanargmin(l[-1])
        dic = dict([(var, val[i,idx]) for var, val in dic.items()])
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
