import scipy as sp
import pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import plots_output
from tqdm import tqdm

FILE = 'AVAL_test.csv'
NB_SER = 15
BATCH_SIZE = 20
df = pd.read_csv(FILE)#.head(NB_SER)
Y = np.array(df['trace'])
X = np.array(df['inputCurrent'])*10 + np.full(Y.shape, 0.001)
T = np.array(df['timeVector'])*1000
DT = T[1] - T[0]

INIT_STATE = [-65]




class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""




    dt = 1
    t = sp.arange(0.0, 450, dt)

    """ The time to  integrate over """

    def __init__(self):
        # build graph
        tf.reset_default_graph()
        self.C_m = tf.get_variable('Cm', initializer=1.0)
        """membrane capacitance, in uF/cm^2"""

        self.g_L = tf.get_variable('g_L', initializer=0.005)
        """Leak maximum conductances, in mS/cm^2"""

        self.E_L = tf.get_variable('E_L', initializer=-50.0)

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t, no_tensor=False):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        if no_tensor:
            return 10 * (t > 100) - 10 * (t > 200) + 35 * (t > 300) - 35 * (t > 400)
        else:
            return 10. * tf.cast((t > 100), tf.float32) - 10. * tf.cast((t > 200), tf.float32) + 35. * tf.cast((t > 300),
                                                            tf.float32) - 35. * tf.cast((t > 400), tf.float32)


    def integ_comp(self, X, i_sin, dt, index=0):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        index += 1

        V = X[0]


        V += ((i_sin - self.I_L(V)) / self.C_m) * dt
        return tf.stack([V], 0), i_sin, dt, index


    def condition(self, hprev, x, dt, index):
        return tf.less(index * dt, DT)


    def step(self, hprev, x):
        div = DT/self.dt
        div= tf.cast(div, tf.float32)
        dt = DT / div
        index = tf.constant(0.)
        h = tf.while_loop(self.condition, self.integ_comp, (hprev, x, dt, index))[0]
        return h

    def step_test(self, X, t):
        return self.integ_comp(X, self.I_inj(t), self.dt)[0]


    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        # inputs
        xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
        ys_ = tf.placeholder(shape=[None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[1], dtype=tf.float32)

        res = tf.scan(self.step,
                      xs_,
                     initializer=init_state)

        V = res[:, 0]
        #V = V * tf.reduce_max(ys_) / tf.reduce_max(V)
        losses = tf.square(tf.subtract(V, ys_))
        loss = tf.reduce_mean(losses)
        loss = tf.Print(loss, [loss], 'loss : ')
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        grads = opt.compute_gradients(loss)
        # capped_grads = capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        train_op = opt.apply_gradients(grads)

        epochs = 200
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0

            results, cacl = sess.run([res, V], feed_dict={
                xs_: X,
                ys_: Y,
                init_state: INIT_STATE
            })
            plots_output(T, X, cacl, Y, suffix=0, show=False, save=True)

            for i in tqdm(range(epochs)):
                final_state = INIT_STATE
                for j in range(0, X.shape[0], BATCH_SIZE):

                    # grad = sess.run(grads, feed_dict={
                    #     xs_: X[j:j + BATCH_SIZE],
                    #     ys_: Y[j:j + BATCH_SIZE],
                    #     init_state: final_state
                    # })
                    # for v, g in grad:
                    #     print(v, g)
                    results, _, train_loss = sess.run([res, train_op, loss], feed_dict={
                        xs_: X[j:j+BATCH_SIZE],
                        ys_: Y[j:j+BATCH_SIZE],
                        init_state: final_state
                    })
                    final_state = results[-1,:]
                    train_loss += train_loss

                # self.plots_output(T, X, cacl, Y)
                print('[{}] loss : {}'.format(i, train_loss))
                train_loss = 0

                results, cacl = sess.run([res, V], feed_dict={
                    xs_: X,
                    ys_: Y,
                    init_state: INIT_STATE
                })
                plots_output(T, X, cacl, Y, suffix=i+1, show=False, save=True)


if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()