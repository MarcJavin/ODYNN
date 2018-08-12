# """
# .. module::
#     :synopsis: Module doing stuff...
#
# .. moduleauthor:: Marc Javin
# """
#
# from . import celeg
# import numpy as np
# import tensorflow as tf
#
# class CElegansCalc(celeg.CElegansNeuron):
#
#     default_init_state = np.append(celeg.CElegansNeuron.default_init_state, [0.])
#     default_params = celeg.CElegansNeuron.default_params.copy()
#     default_params['k_b'] = 1.
#     default_params['k_f'] = 0.5
#     default_params['B'] = 30.
#     default_params['G_pump'] = 3.
#     default_params['K_pump'] = 70.9
#     # del default_params['decay_ca']
#
#     _constraints_dic = celeg.CElegansNeuron._constraints_dic.copy()
#     _constraints_dic['k_b'] = [10e-3, np.infty]
#     _constraints_dic['k_f'] = [10e-3, np.infty]
#     _constraints_dic['B'] = [10e-3, np.infty]
#     _constraints_dic['G_pump'] = [10e-3, np.infty]
#     _constraints_dic['K_pump'] = [10e-3, np.infty]
#     # del _constraints_dic['decay_ca']
#
#     def __init__(self, init_p=None, tensors=False, dt=0.1):
#         celeg.CElegansNeuron.__init__(self, init_p=init_p, tensors=tensors, dt=dt)
#
#     @staticmethod
#     def get_random():
#         rand = celeg.CElegansNeuron.get_random()
#         rand['k_b'] = np.random.uniform(10e-3, 10.)
#         rand['k_f'] = np.random.uniform(10e-3, 10.)
#         rand['B'] = np.random.uniform(1., 10000.)
#         rand['G_pump'] = np.random.uniform(0.1, 1000.)
#         rand['K_pump'] = np.random.uniform(10e-2, 100.)
#         # del rand['decay_ca']
#         return rand
#
#     def step(self, X, i_inj):
#         V = X[self.V_pos]
#         p = X[1]
#         q = X[2]
#         n = X[3]
#         e = X[4]
#         f = X[5]
#         cac = X[-2]
#         cacb = X[-1]
#
#         h = self._h(cac)
#         V = (V*(self._param['C_m']/self.dt) + (i_inj + self.g_Ca(e,f,h)*self._param['E_Ca'] + (self.g_Ks(n)+self.g_Kf(p,q))*self._param['E_K'] + self._param['g_L']*self._param['E_L'])) / \
#             ((self._param['C_m']/self.dt) + self.g_Ca(e,f,h) + self.g_Ks(n) + self.g_Kf(p,q) + self._param['g_L'])
#
#         kb = self._param['k_b']
#         B = self._param['B']
#         kf = self._param['k_f']
#         Gp = self._param['G_pump']
#         Kp = self._param['K_pump']
#         cac = (cac - self._param['rho_ca'] * self._i_ca(V,e,f,h) + kb*B*cacb) / (1 + kf*B*(1-cacb) - Gp/(cac + Kp))
#         # cac = cac - self._param['rho_ca'] * self._i_ca(V,e,f,h) + kb*B*cacb - kf*cac*B*(1-cacb) - (Gp*cac)/(cac + Kp)
#         cacb = (cacb + kf*cac) / (1 + kb + kf*cac)
#         # cacb = cacb - kb*cacb + kf*cac*(1-cacb)
#         p = self._update_gate(p, 'p', V)
#         q = self._update_gate(q, 'q', V)
#         n = self._update_gate(n, 'n', V)
#         e = self._update_gate(e, 'e', V)
#         f = self._update_gate(f, 'f', V)
#
#         if self._tensors:
#             return tf.stack([V, p, q, n, e, f, cac, cacb], 0)
#         else:
#             return np.array([V, p, q, n, e, f, cac, cacb])