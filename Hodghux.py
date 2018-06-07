import tensorflow as tf
import scipy as sp
import params
import numpy as np



class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""


    REST_CA = params.REST_CA

    def __init__(self, init_p=params.DEFAULT, tensors=False, loop_func=None, dt=0.1, fixed=None, constraints=None):
        self.tensors = tensors
        self.param = init_p
        self.inits_p = {self.ik_from_v: params.INIT_STATE_ik,
                   self.ica_from_v: params.INIT_STATE_ica}
        if (loop_func is not None):
            self.loop_func = loop_func
        self.init_state = self.get_init_state()
        self.dt = dt


    def inf(self, V, rate):
        mdp = self.param['%s__mdp' % rate]
        scale = self.param['%s__scale' % rate]
        if(self.tensors):
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

    @staticmethod 
    def integ_comp(X, i_inj, self):
        """
        Integrate
        """
        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]

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

    @staticmethod 
    def no_tau_ca(X, i_inj, self):
        """
        Integrate
        """
        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]
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
        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]
        h = self.h(cac)
        V += ((i_inj - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(
            V)) / self.param['C_m']) * self.dt

        cac += (-self.I_Ca(V, e, f, h) * self.param['rho_ca'] - (
                    (cac - self.REST_CA) / self.param['decay_ca'])) * self.dt
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
        cac = X[-1]

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

    loop_func = integ_comp



    def get_init_state(self):
        return self.inits_p.get(self.loop_func, params.INIT_STATE)

class Neuron_tf(HodgkinHuxley):

    nb=-1

    def __init__(self, init_p=params.DEFAULT, loop_func=None, dt=0.1, fixed=[], constraints=params.CONSTRAINTS):
        HodgkinHuxley.__init__(self, init_p=init_p, tensors=True, loop_func=loop_func, dt=dt)
        self.init_p = init_p
        self.fixed = fixed
        self.constraints_dic = constraints
        self.id = self.give_id()

    @classmethod
    def give_id(cls):
        cls.nb +=1
        return str(cls.nb)

    def step(self, hprev, x):
        return self.loop_func(hprev, x, self)

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

class Neuron_set_tf(Neuron_tf):

    """Set of neurons to perform operations on vectors"""

    def __init__(self, inits_p, loop_func=None, fixed=params.ALL, constraints=params.CONSTRAINTS, dt=0.1):
        #dictionnary with values in arrays
        init_p = dict([(var, [p[var] for p in inits_p]) for var in inits_p[0].iterkeys()])
        Neuron_tf.__init__(self, init_p=init_p, loop_func=loop_func, dt=dt, fixed=fixed, constraints=constraints)

        self.num = len(inits_p)
        self.init_state = np.tile(self.init_state, (len(inits_p),1)).transpose()
        print(self.init_state.shape)


class Neuron_fix(HodgkinHuxley):

    def __init__(self, init_p=params.DEFAULT, loop_func=None, dt=0.1):
        HodgkinHuxley.__init__(self, init_p=init_p, tensors=False, loop_func=loop_func, dt=dt)
        self.state = self.init_state

    def get_volt(self):
        return self.state[0]

    def step(self, i):
        self.state = np.array(self.loop_func(self.state, i, self))
        return self.state

    def reset(self):
        self.state = self.init_state

class Neuron_set_fix(Neuron_fix):
    def __init__(self, inits_p, loop_func=None, dt=0.1):
        # dictionnary with values in arrays
        init_p = dict([(var, np.array([p[var] for p in inits_p])) for var in inits_p[0].iterkeys()])
        Neuron_fix.__init__(self, init_p=init_p, loop_func=loop_func, dt=dt)
        self.num = len(inits_p)

        self.init_state = np.tile((self.init_state), (len(inits_p),1)).transpose()
        self.state = self.init_state