import tensorflow as tf
import scipy as sp
import neuron_params
import numpy as np
from abc import ABC, abstractmethod

V_pos = 0
Ca_pos = -1

class Model(ABC):

    default = None
    constraints_dic = None
    init_state = None

    def __init__(self, init_p=neuron_params.DEFAULT, tensors=False, dt=0.1):
        self.tensors = tensors
        if (isinstance(init_p, list)):
            self.num = len(init_p)
            init_p = dict([(var, np.array([p[var] for p in init_p], dtype=np.float32)) for var in init_p[0].keys()])
            self.init_state = np.stack([self.init_state for _ in range(self.num)], axis=1)
        else:
            self.num = 1
        self.param = init_p
        self.dt = dt

    @staticmethod
    @abstractmethod
    def step_model(X, i, self):
        pass

    @staticmethod
    @abstractmethod
    def get_random():
        pass

    def get_init_state(self):
        return self.init_state

class HodgkinHuxley(Model):
    """Full Hodgkin-Huxley Model implemented in Python"""


    REST_CA = neuron_params.REST_CA
    init_state = neuron_params.INIT_STATE
    default = neuron_params.DEFAULT
    constraints_dic = neuron_params.CONSTRAINTS

    def __init__(self, init_p=neuron_params.DEFAULT, tensors=False, dt=0.1):
        Model.__init__(self, init_p=init_p, tensors=tensors, dt=dt)

    """steady state value of a rate"""
    def inf(self, V, rate):
        mdp = self.param['%s__mdp' % rate]
        scale = self.param['%s__scale' % rate]
        if(self.tensors):
            # print('V : ', V)
            # print('mdp : ', mdp)
            return tf.sigmoid((V - mdp) / scale)
        else:
            return 1 / (1 + sp.exp((mdp - V)/scale))

    def h(self, cac):
        """Channel gating kinetics. Functions of membrane voltage"""
        q = self.inf(cac, 'h')
        return 1 + (q - 1) * self.param['h__alpha']

    def g_Ca(self, e, f, h):
        return self.param['g_Ca'] * e ** 2 * f * h

    def I_Ca(self, V, e, f, h):
        """
        Membrane current (in uA/cm^2)
        Calcium (Ca = element name)
        """
        return self.param['g_Ca'] * e ** 2 * f * h * (V - self.param['E_Ca'])

    def g_Kf(self, p, q):
        return self.param['g_Kf'] * p ** 4 * q

    def I_Kf(self, V, p, q):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.param['g_Kf'] * p ** 4 * q * (V - self.param['E_K'])

    def g_Ks(self, n):
        return self.param['g_Ks'] * n

    def I_Ks(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.param['g_Ks'] * n * (V - self.param['E_K'])

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak
        """
        return self.param['g_L'] * (V - self.param['E_L'])

    """default model"""
    @staticmethod
    def step_model(X, i_inj, self):
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
        # V = V * (i_inj + self.g_Ca(e,f,h)*self.param['E_Ca'] + (self.g_Ks(n)+self.g_Kf(p,q))*self.param['E_K'] + self.param['g_L']*self.param['E_L']) / \
        #     ((self.param['C_m']/self.dt) + self.g_Ca(e,f,h) + self.g_Ks(n) + self.g_Kf(p,q) + self.param['g_L'])
        V += ((i_inj - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self.param[
            'C_m']) * self.dt


        cac += (-self.I_Ca(V, e, f, h) * self.param['rho_ca'] - ((cac - self.REST_CA) / self.param['decay_ca'])) * self.dt
        tau = self.param['p__tau']
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self.inf(V, 'p') / tau))
        tau = self.param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self.inf(V, 'q') / tau))
        tau = self.param['e__tau']
        e = ((tau * self.dt) / (tau + self.dt)) * ((e / self.dt) + (self.inf(V, 'e') / tau))
        tau = self.param['f__tau']
        f = ((tau * self.dt) / (tau + self.dt)) * ((f / self.dt) + (self.inf(V, 'f') / tau))
        tau = self.param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self.inf(V, 'n') / tau))

        if(self.tensors):
            return tf.stack([V, p, q, n, e, f, cac], 0)
        else:
            return [V, p, q, n, e, f, cac]

    def get_random(self):
        return neuron_params.give_rand()

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
            V)) / self.param['C_m']) * self.dt
        cac += (-self.I_Ca(V, e, f, h) * self.param['rho_ca'] - (
                    (cac - self.REST_CA) / self.param['decay_ca'])) * self.dt
        tau = self.param['p__tau']
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self.inf(V, 'p') / tau))
        tau = self.param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self.inf(V, 'q') / tau))
        tau = self.param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self.inf(V, 'n') / tau))
        e = self.inf(V, 'e')
        f = self.inf(V, 'f')
        if (self.tensors):
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
            V)) / self.param['C_m']) * self.dt
        # cac = (self.param['decay_ca'] / (self.dt + self.param['decay_ca'])) * (
        #             cac - self.I_Ca(V, e, f, h) * self.param['rho_ca'] * self.dt + self.REST_CA * self.param['decay_ca'] / self.dt)
        # 
        # cac += (-self.I_Ca(V, e, f, h) * self.param['rho_ca'] - (
        #             (cac - self.REST_CA) / self.param['decay_ca'])) * self.dt
        p = self.inf(V, 'p')
        q = self.inf(V, 'q')
        e = self.inf(V, 'e')
        f = self.inf(V, 'f')
        n = self.inf(V, 'n')
        if (self.tensors):
            return tf.stack([V, p, q, n, e, f, cac], 0)
        else:
            return [V, p, q, n, e, f, cac]

    @staticmethod 
    def ica_from_v(X, v_fix, self):
        e = X[1]
        f = X[2]
        cac = X[Ca_pos]

        h = self.h(cac)
        tau = self.param['e__tau']
        e = ((tau * self.dt) / (tau + self.dt)) * ((e / self.dt) + (self.inf(v_fix, 'e') / tau))
        tau = self.param['f__tau']
        f = ((tau * self.dt) / (tau + self.dt)) * ((f / self.dt) + (self.inf(v_fix, 'f') / tau))
        ica = self.I_Ca(v_fix, e, f, h)
        cac += (-self.I_Ca(v_fix, e, f, h) * self.param['rho_ca'] - (
                    (cac - self.REST_CA) / self.param['decay_ca'])) * self.dt

        if (self.tensors):
            return tf.stack([ica, e, f, h, cac], 0)
        else:
            return [ica, e, f, h, cac]

    @staticmethod
    def ik_from_v(X, v_fix, self):
        p = X[1]
        q = X[2]
        n = X[3]

        tau = self.param['p__tau']
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self.inf(v_fix, 'p') / tau))
        tau = self.param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self.inf(v_fix, 'q') / tau))
        tau = self.param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self.inf(v_fix, 'n') / tau))
        ik = self.I_Kf(v_fix, p, q) + self.I_Ks(v_fix, n)

        if (self.tensors):
            return tf.stack([ik, p, q, n], 0)
        else:
            return [ik, p, q, n]


MODEL = HodgkinHuxley

class NeuronTf(MODEL):

    nb=-1

    def __init__(self, init_p=neuron_params.DEFAULT, loop_func=None, dt=0.1, fixed=[], constraints=neuron_params.CONSTRAINTS):
        HodgkinHuxley.__init__(self, init_p=init_p, tensors=True, dt=dt)
        self.init_p = self.param
        self.fixed = fixed
        self.constraints_dic = constraints
        self.id = self.give_id()

    @classmethod
    def give_id(cls):
        cls.nb +=1
        return str(cls.nb)

    def step(self, hprev, x):
        return self.step_model(hprev, x, self)

    """rebuild tf variable graph"""
    def reset(self):
        with(tf.variable_scope(self.id)):
            self.param = {}
            self.constraints = []
            for var, val in self.init_p.items():
                if (var in self.fixed):
                    self.param[var] = tf.constant(val, name=var, dtype=tf.float32)
                else:
                    self.param[var] = tf.get_variable(var, initializer=val, dtype=tf.float32)
                    if var in self.constraints_dic:
                        con = self.constraints_dic[var]
                        self.constraints.append(
                            tf.assign(self.param[var], tf.clip_by_value(self.param[var], con[0], con[1])))
        # print('neuron_params after reset : ', self.param)

    def parallelize(self, n):
        """Add a dimension of size n in the parameters"""
        if(self.num > 1):
            self.init_p = dict([(var, np.stack([val for _ in range(n)], axis=val.ndim)) for var, val in self.init_p.items()])
        else:
            self.init_p = dict(
                [(var, np.stack([val for _ in range(n)], axis=0)) for var, val in self.init_p.items()])
        self.init_state = np.stack([self.init_state for _ in range(n)], axis=self.init_state.ndim)


    def build_graph(self, batch=None):
        tf.reset_default_graph()
        self.reset()
        xshape = [None]
        initializer = self.init_state
        if (batch is not None):
            xshape.append(None)
            initializer = np.stack([initializer for _ in range(batch)], axis=1)
        if (self.num > 1):
            xshape.append(self.num)
        curs_ = tf.placeholder(shape=xshape, dtype=tf.float32, name='input_current')
        res_ = tf.scan(self.step,
                      curs_,
                      initializer=initializer.astype(np.float32))
        return curs_, res_


    def calculate(self, i):
        if(i.ndim > 1):
            input_cur, res_ = self.build_graph(batch=i.shape[1])
        else:
            input_cur, res_ = self.build_graph()
        if(i.ndim < 3 and self.num > 1):
            i = np.stack([i for _ in range(self.num)], axis=i.ndim)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run(res_, feed_dict={
                input_cur: i
            })
        return results



class NeuronFix(MODEL):

    def __init__(self, init_p=neuron_params.DEFAULT, dt=0.1):
        HodgkinHuxley.__init__(self, init_p=init_p, tensors=False, dt=dt)
        self.state = self.init_state

    def get_volt(self):
        return self.state[V_pos]

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

    def reset(self, init_p=None):
        self.state = self.init_state