import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from utils import plots_output, plots_results_ca, plots_results, get_data
from params import PARAM_MEMB, PARAM_GATES
import params
from tqdm import tqdm
import time

FILE = 'AVAL_test.csv'
NB_SER = 15
BATCH_SIZE = 60
T, X, Y = get_data()
Y = Y*75 - 65
X = X*2
DT = T[1] - T[0]

INIT_STATE = [-65, 0, 1, 0]
INIT_STATE = [-65, 0., 0.95, 0, 0, 1, 1e-7]





class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    DECAY_CA = 11.6  # ms
    RHO_CA = 0.000239e3  # mol_per_cm_per_uA_per_ms
    REST_CA = 0  # M

    dt = 0.1
    t = params.t
    i_inj = params.i_inj



    """ The time to  integrate over """

    def __init__(self):
        # build graph
        tf.reset_default_graph()
        self.C_m = tf.get_variable('Cm', initializer=PARAM_MEMB['C_m'])

        self.memb = {}
        for var, val in PARAM_MEMB.items():
            self.memb[var] = tf.get_variable(var, initializer=val)

        self.rates = {}
        for var, val in PARAM_GATES.items():
            self.rates[var] = tf.get_variable(var, initializer=val)

    def inf(self, V, rate):
        mdp = self.rates['%s__mdp' % rate]
        scale = self.rates['%s__scale' % rate]
        return tf.sigmoid((V - mdp) / scale)

    def h(self, cac):
        """Channel gating kinetics. Functions of membrane voltage"""
        mdp = self.rates['h__mdp']
        scale = self.rates['h__scale']
        alpha = self.rates['h__alpha']
        q = tf.sigmoid((cac - mdp) / scale)
        return 1 + (q - 1) * alpha


    def I_Ca(self, V, e, f, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)
        """
        return self.memb['g_Ca'] * e ** 2 * f * h * (V - self.memb['E_Ca'])

    def I_Kf(self, V, p, q):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.memb['g_Kf'] * p ** 4 * q * (V - self.memb['E_K'])

    def I_Ks(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.memb['g_Ks'] * n * (V - self.memb['E_K'])

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak
        """
        return self.memb['g_L'] * (V - self.memb['E_L'])


    def integ_comp(self, X, i_sin, dt, index=0):
        """
        Integrate
        """
        index += 1

        V = X[0]
        e = X[-3]
        f = X[-2]
        cac = X[-1]

        h = self.h(cac)
        V += ((i_sin - self.I_Ca(V, e, f, h) - self.I_L(V)) / self.C_m) * dt
        cac = (self.DECAY_CA / (dt + self.DECAY_CA)) * (cac - self.I_Ca(V, e, f, h) * self.RHO_CA * dt + self.REST_CA * self.DECAY_CA / dt)
        tau = self.rates['e__tau']
        e = ((tau * dt) / (tau + dt)) * ((e / dt) + (self.inf(V, 'e') / tau))
        tau = self.rates['f__tau']
        f = ((tau * dt) / (tau + dt)) * ((f / dt) + (self.inf(V, 'f') / tau))
        return tf.stack([V, e, f, cac], 0), i_sin, dt, index


    def integ_complete(self, X, i_sin, dt, index=0):
        """
        Integrate
        """
        index += 1

        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]


        h = self.h(cac)
        V += ((i_sin - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self.C_m) * dt
        cac = (self.DECAY_CA/(dt+self.DECAY_CA)) * (cac - self.I_Ca(V, e, f, h)*self.RHO_CA*dt + self.REST_CA*self.DECAY_CA/dt)
        tau = self.rates['p__tau']
        p = ((tau*dt) / (tau+dt)) * ((p/dt) + (self.inf(V, 'p')/tau))
        tau = self.rates['q__tau']
        q = ((tau*dt) / (tau+dt)) * ((q/dt) + (self.inf(V, 'q')/tau))
        tau = self.rates['e__tau']
        e = ((tau*dt) / (tau+dt)) * ((e/dt) + (self.inf(V, 'e')/tau))
        tau = self.rates['f__tau']
        f = ((tau*dt) / (tau+dt)) * ((f/dt) + (self.inf(V, 'f')/tau))
        tau = self.rates['n__tau']
        n = ((tau*dt) / (tau+dt)) * ((n/dt) + (self.inf(V, 'n')/tau))
        return tf.stack([V, p, q, n, e, f, cac], 0), i_sin, dt, index

    def no_tau(self, X, i_sin, dt, index=0):
        """
        Integrate
        """
        index += 1
        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]
        h = self.h(cac)
        V += ((i_sin - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(
            V)) / self.C_m) * dt
        cac += (-self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)) * dt
        cac = (self.DECAY_CA / (dt + self.DECAY_CA)) * (
                    cac - self.I_Ca(V, e, f, h) * self.RHO_CA * dt + self.REST_CA * self.DECAY_CA / dt)
        p = self.inf(V, 'p')
        q = self.inf(V, 'q')
        e = self.inf(V, 'e')
        f = self.inf(V, 'f')
        n = self.inf(V, 'n')
        return tf.stack([V, p, q, n, e, f, cac], 0), i_sin, dt, index

    loop_func = integ_complete


    def condition(self, hprev, x, dt, index):
        return tf.less(index * dt, DT)


    def step(self, hprev, x):
        div = DT/self.dt
        div= tf.cast(div, tf.float32)
        dt = DT / div
        index = tf.constant(0.)
        h = tf.while_loop(self.condition, self.loop_func, (hprev, x, dt, index))[0]
        return h

    def step_test(self, X, i):
        return self.loop_func(X, i, self.dt)[0]

    def test(self):

        ts_ = tf.placeholder(shape=[None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[7], dtype=tf.float32)

        res = tf.scan(self.step_test,
                      self.i_inj,
                      initializer=init_state)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            results = sess.run(res, feed_dict={
                ts_: self.t,
                init_state: INIT_STATE
            })
            print('time spent : %.2f s' % (time.time() - start))
        plots_results(self, self.t, self.i_inj, results, cur=False)


    def Main(self, prefix):
        # inputs
        xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
        ys_ = tf.placeholder(shape=[None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[7], dtype=tf.float32)

        res = tf.scan(self.step,
                      xs_,
                     initializer=init_state)

        cac = res[:, 0]
        #V = V * tf.reduce_max(ys_) / tf.reduce_max(V)
        losses = tf.square(tf.subtract(cac, ys_))
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

            for i in tqdm(range(epochs)):
                results, cacl, _, train_loss = sess.run([res, cac, train_op, loss], feed_dict={
                    xs_: X,
                    ys_: Y,
                    init_state: INIT_STATE
                })
                train_loss += train_loss

                # self.plots_output(T, X, cacl, Y)
                print('[{}] loss : {}'.format(i, train_loss))
                train_loss = 0

                plots_output(T, X, cacl, Y, suffix='%s_%s'%(prefix,i + 1), show=False, save=True)
                # plots_results_ca(self, T, X, results, suffix=0, show=False, save=True)


if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.loop_func = runner.no_tau
    runner.Main('notau')
