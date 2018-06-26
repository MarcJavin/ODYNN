import tensorflow as tf
from utils import OUT_SETTINGS, set_dir, OUT_PARAMS, plot_loss_rate
from data import FILE_LV, SAVE_PATH, get_data_dump
import pickle
import numpy as np
import time
from abc import ABC, abstractmethod



class Optimized(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def settings(self):
        pass

    @staticmethod
    def plot_vars(var_dic, suffix, show, save):
        pass

    def apply_constraints(self, session):
        pass

    def get_params(self):
        return []


class Optimizer(ABC):

    min_loss = 1.

    def __init__(self, optimized):
        self.start_time = time.time()
        self.optimized = optimized
        self.parallel = self.optimized.num

    """learning rate and optimization"""
    def build_train(self):
        self.global_step = tf.Variable(0, trainable=False)
        # progressive learning rate
        self.learning_rate = tf.train.exponential_decay(
            self.start_rate,  # Base learning rate.
            self.global_step,  # Current index to the dataset.
            self.decay_step,  # Decay step.
            self.decay_rate,  # Decay rate.
            staircase=True)
        # self.learning_rate = 0.1
        tf.summary.scalar('learning rate', self.learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gvs = opt.compute_gradients(self.loss)
        grads, vars = zip(*gvs)
        print(grads)

        if(self.parallel > 1):
            grads_normed = []
            for i in range(self.parallel):
                #clip by norm for each parallel model (neuron or circuit)
                gi = [g[..., i] for g in grads]
                # if(isinstance(self.optimized, Circuit.Circuit)):
                #     #[synapse, model]
                #     gi = [g[:,i] for g in grads]
                # else:
                #     gi = [g[i] for g in grads]
                gi_norm, _ = tf.clip_by_global_norm(gi, 5.)
                grads_normed.append(gi_norm)
            grads_normed = tf.stack(grads_normed)
            #resize to tf format
            try: #for circuits
                grads_normed = tf.transpose(grads_normed, perm=[1,2,0])
                grads_normed = tf.unstack(grads_normed, axis=0)
            except:
                grads_normed = tf.unstack(grads_normed, axis=1)
        else:
            grads_normed, _ = tf.clip_by_global_norm(grads, 5.)
        self.train_op = opt.apply_gradients(zip(grads_normed, vars), global_step=self.global_step)

        self.saver = tf.train.Saver()

    """initialize objects to be optimized and write setting in the directory"""
    def init(self, subdir, suffix, file, l_rate, w, yshape):
        self.suffix = suffix
        self.dir = set_dir(subdir + '/')
        tf.reset_default_graph()
        self.start_rate, self.decay_step, self.decay_rate = l_rate
        self.write_settings(w)

        self.T, self.X, self.V, self.Ca = get_data_dump(file)
        assert (self.optimized.dt == self.T[1] - self.T[0])
        self.n_batch = self.X.shape[1]


        if (self.parallel > 1):
            # add dimension for neurons trained in parallel
            # [time, n_batch, neuron]
            self.X = np.stack([self.X for _ in range(self.parallel)], axis=self.X.ndim)
            self.V = np.stack([self.V for _ in range(self.parallel)], axis=self.V.ndim)
            self.Ca = np.stack([self.Ca for _ in range(self.parallel)], axis=self.Ca.ndim)
            yshape.append(self.parallel)

        self.xs_, self.res = self.optimized.build_graph(batch=self.n_batch)
        self.ys_ = tf.placeholder(shape=yshape, dtype=tf.float32, name='voltage_Ca')

        print('i expected : ', self.xs_.shape)
        print('i : ', self.X.shape, 'V : ', self.V.shape)


    def write_settings(self, w):
        with open(self.dir + OUT_SETTINGS, 'w') as f:
            f.write('Weights (out, cac) : %s' % w + '\n' +
                'Start rate : %s, decay_step : %s, decay_rate : %s' % (
                self.start_rate, self.decay_step, self.decay_rate) + '\n' +
                    self.optimized.settings())

    """train the model and collect loss, learn_rate and variables"""
    def train_and_gather(self, sess, i, losses, rates, vars):
        summ, results, _, train_loss = sess.run([self.summary, self.res, self.train_op, self.loss], feed_dict={
            self.xs_: self.X,
            self.ys_: np.array([self.V, self.Ca])
        })

        self.tdb.add_summary(summ, i)

        self.optimized.apply_constraints(sess)

        with open('%s%s_%s.txt' % (self.dir, OUT_PARAMS, self.suffix), 'w') as f:
            for name, v in self.optimized.get_params():
                v_ = sess.run(v)
                f.write('%s : %s\n' % (name, v_))
                vars[name][i + 1] = v_

        rates[i] = sess.run(self.learning_rate)
        losses[i] = train_loss
        if(self.parallel > 1):
            train_loss = np.nanmean(train_loss)
        print('[{}] loss : {}'.format(i, train_loss))
        return results

    def plots_dump(self, sess, losses, rates, vars, i):
        with (open(self.dir + FILE_LV, 'wb')) as f:
            pickle.dump([losses, rates, vars], f)
        plot_loss_rate(losses[:i + 1], rates[:i + 1], suffix=self.suffix, show=False, save=True)
        self.saver.save(sess, '%s%s' % (self.dir, SAVE_PATH))
        self.optimized.plot_vars(dict([(name, val[:i + 2]) for name, val in vars.items()]), suffix=self.suffix+'evolution', show=False,
                  save=True)