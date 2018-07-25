"""
.. module:: neuron_training
    :synopsis: Module containing functions to organize the training of neurons

.. moduleauthor:: Marc Javin
"""
import sys

import numpy as np
import scipy as sp

from opthh import utils, cfg_model, hhmodel, datas, optim
from opthh.neuron import NeuronLSTM, BioNeuronTf
from opthh.noptim import NeuronOpt
from opthh import nsimul as sim

CA_VAR = {'e__tau', 'e__mdp', 'e__scale', 'f__tau', 'f__mdp', 'f__scale', 'h__alpha', 'h__mdp', 'h__scale', 'g_Ca',
          'E_Ca', 'rho_ca', 'decay_ca'}
K_VAR = {'p__tau', 'p__mdp', 'p__scale', 'q__tau', 'q__mdp', 'q__scale', 'n_tau', 'n__mdp', 'n__scale', 'g_Kf', 'g_Ks',
         'E_K'}

CA_CONST = hhmodel.ALL - CA_VAR
K_CONST = hhmodel.ALL - K_VAR

MODEL = cfg_model.NEURON_MODEL

pars = [MODEL.get_random() for i in range(10)]
# pars = data.get_vars('Init_settings_100_2', 0)
# pars = [dict([(ki, v[n]) for k, v in pars.items()]) for n in range(len(pars['C_m']))]
dt = 0.2
t, iinj = datas.give_train(dt)
i_inj = iinj
tt, it = datas.give_test(dt)
"""Single optimisation"""


def single_exp(xp, w_v, w_ca, suffix=None):
    name = 'Classic'

    opt = NeuronOpt()
    base = MODEL.step_model

    if (xp == 'ica'):
        name = 'Icafromv'
        opt = NeuronOpt(BioNeuronTf(fixed=CA_CONST))
        loop_func = MODEL.ica_from_v

    elif (xp == 'ik'):
        name = 'Ikfromv'
        opt = NeuronOpt(BioNeuronTf(fixed=K_CONST))
        loop_func = MODEL.ik_from_v

    elif (xp == 'notauca'):
        name = 'Notauca'
        loop_func = MODEL.no_tau_ca

    elif (xp == 'notau'):
        name = 'Notau'
        loop_func = MODEL.no_tau

    elif (xp == 'classic'):
        name = 'integcomp'
        loop_func = base

    print(name, w_v, w_ca, loop_func)
    dir = '%s_v=%s_ca=%s' % (name, w_v, w_ca)
    if (suffix is not None):
        dir = '%s_%s' % (dir, suffix)
    dir = utils.set_dir(dir)
    MODEL.step_model = loop_func
    MODEL.step_model = loop_func
    train = sim.simul(dt=dt, i_inj=i_inj, show=True)
    opt.optimize(dir, w=[w_v, w_ca], train=train)
    MODEL.step_model = base
    return dir


def steps2_exp_ca(w_v1, w_ca1, w_v2, w_ca2):
    name = '_2steps'

    dir = single_exp('ica', w_v1, w_ca1, suffix='%s%s%s' % (name, w_v2, w_ca2))

    param = optim.get_best_result(dir)
    opt = NeuronOpt(BioNeuronTf(init_p=param, fixed=CA_VAR))
    train = sim.simul(p=MODEL.default_params, dt=dt, i_inj=i_inj, suffix='step2', show=False)
    opt.optimize(dir, w=[w_v2, w_ca2], l_rate=[0.1, 9, 0.9], suffix='step2', train=train)

    test_xp(dir)


def steps2_exp_k(w_v2, w_ca2):
    name = '_2steps'

    dir = single_exp('ik', 1, 0, suffix='%s%s%s' % (name, w_v2, w_ca2))

    param = optim.get_best_result(dir)
    opt = NeuronOpt(BioNeuronTf(init_p=param, fixed=K_VAR))
    train = sim.simul(dt=dt, i_inj=i_inj, suffix='step2')
    opt.optimize(dir, w=[w_v2, w_ca2], l_rate=[0.1, 9, 0.9], suffix='step2', train=train)

    test_xp(dir)


def test_xp(dir, i=i_inj, default=MODEL.default_params, suffix='', show=False):
    dir = utils.set_dir(dir)
    param = optim.get_best_result(dir)
    for j, i_ in enumerate(i.transpose()):
        sim.comp_pars_targ(param, default, dt=dt, i_inj=i_, show=show, save=True, suffix='train%s' % j)

    dt2 = 0.05
    tt = np.array(sp.arange(0.0, 4000, dt2))
    t3 = np.array(sp.arange(0.0, 6000, dt2))
    i1 = (tt - 1000) * (30. / 200) * ((tt > 1000) & (tt <= 1200)) + 30 * ((tt > 1200) & (tt <= 3000)) - (tt - 2800) * (
                30. / 200) * ((tt > 2800) & (tt <= 3000))
    i2 = (tt - 1000) * (50. / 1000) * ((tt > 1000) & (tt <= 2000)) + (3000 - tt) * (50. / 1000) * (
                (tt > 2000) & (tt <= 3000))
    i3 = (t3 - 1000) * (1. / 2000) * ((t3 > 1000) & (t3 <= 3000)) + (5000 - t3) * (1. / 2000) * (
                (t3 > 3000) & (t3 <= 5000))
    is_ = [i1, i2, i3]
    ts_ = [tt, tt, t3]
    for j, i_ in enumerate(is_):
        sim.comp_pars_targ(param, default, dt=dt2, i_inj=i_, show=show, save=True, suffix='test%s' % j)


def alternate(name='', suffix='', lstm=True):
    dir = 'Integcomp_alternate_%s' % name
    wv = 1
    wca = 0
    if (lstm):
        dir += '_lstm'
        neur = NeuronLSTM(dt=dt)
        l_rate = [0.01, 9, 0.95]
    else:
        neur = BioNeuronTf(pars, dt=dt)
        l_rate = [1., 9, 0.92]
    opt = NeuronOpt(neur)
    dir = utils.set_dir(dir)
    train = sim.simul(dt=dt, i_inj=i_inj, show=False, suffix='train')
    opt.optimize(dir, suffix=suffix, train=train, w=[wv, wca], epochs=300, step=0, l_rate=l_rate)
    for i in range(40):
        wv -= 1./50
        wca += 1./50
        n = opt.optimize(dir, suffix=suffix, train=train, w=[wv, wca], epochs=10, l_rate=l_rate, reload=True, step=i + 1)
    test_xp(dir)


def classic(name, wv, wca, default=MODEL.default_params, suffix='', lstm=False):
    if (wv == 0):
        dir = 'Integcomp_calc_%s' % name
    elif (wca == 0):
        dir = 'Integcomp_volt_%s' % name
    else:
        dir = 'Integcomp_both_%s' % name
    if (lstm):
        dir += '_lstm'
        neur = NeuronLSTM(dt=dt)
        l_rate = [0.01, 9, 0.95]
        opt = NeuronOpt(neur)
    else:
        neur = BioNeuronTf(pars, dt=dt)
        l_rate = [1., 9, 0.92]
        opt = NeuronOpt(neur)
    dir = utils.set_dir(dir)
    train = sim.simul(p=default, dt=dt, i_inj=i_inj, show=False, suffix='train')
    test= sim.simul(p=default, dt=dt, i_inj=it, show=False, suffix='test')
    n = opt.optimize(dir, w=[wv, wca], train=train, test=test, suffix=suffix, l_rate=l_rate, evol_var=False)#, reload=True, reload_dir='Integcomp_both_incr1-0_lstm-YAY')
    test_xp(dir, default=default)


def real_data(name, suffix='', lstm=True):
    dir = 'Real_data_%s' % name
    if (lstm):
        dir += '_lstm'
        neur = NeuronLSTM(dt=dt)
        l_rate = [0.01, 9, 0.95]
        opt = NeuronOpt(neur)
    else:
        neur = BioNeuronTf(pars, dt=dt)
        l_rate = [1., 9, 0.92]
        opt = NeuronOpt(neur)
    dir = utils.set_dir(dir)
    train, test = datas.dump_data(dt=dt)
    n = opt.optimize(dir, w=[0, 1], train = train, suffix=suffix, l_rate=l_rate)
    comp_pars(dir, n)
    t, i, v, ca = test
    if not lstm:
        sim.simul(optim.get_best_result(dir), dt=dt, i_inj=i_inj, suffix='test', save=True, ca_true=ca)


def comp_pars(dir, i=-1):
    dir = utils.set_dir(dir)
    p = optim.get_vars(dir, i)
    hhmodel.plot_vars(p, func=utils.bar, suffix='compared', show=False, save=True)
    hhmodel.boxplot_vars(p, suffix='boxes', show=False, save=True)


def add_plots():
    import glob
    import re
    for filename in glob.iglob(utils.RES_DIR + '*'):
        dir = re.sub(utils.RES_DIR, '', filename)
        try:
            comp_pars(dir)
        except:
            print(dir)

def test_lstm():
    import pickle
    with open('data/optimized4', 'rb') as f:
        load = pickle.load(f)
    n = NeuronLSTM(load=load)
    trace, test = datas.dump_data(dt=n.dt)
    trace = np.array(test[2:])
    sim.comp_neuron_trace(n, trace, i_inj=test[1], scale=True)
    exit(0)

if __name__ == '__main__':

    xp = sys.argv[1]
    if len(sys.argv)>3:
        suf = sys.argv[3]
    else:
        suf = ''
    if (xp == 'alt'):
        name = sys.argv[2]
        alternate(name, suffix=suf)
    elif (xp == 'cac'):
        name = sys.argv[2]
        classic(name, wv=0, wca=1, suffix=suf)
    elif (xp == 'v'):
        name = sys.argv[2]
        classic(name, wv=1, wca=0, suffix=suf)
    elif (xp == 'both'):
        name = sys.argv[2]
        classic(name, wv=1, wca=1, suffix=suf)
    elif (xp == 'real'):
        name = sys.argv[2]
        real_data(name, suffix=suf)
    elif (xp == 'single'):
        xp = sys.argv[2]
        w_v, w_ca = list(map(int, sys.argv[3:5]))
        single_exp(xp, w_v, w_ca)
    elif (xp == '2stepsca'):
        w_v1, w_ca1, w_v2, w_ca2 = list(map(int, sys.argv[2:6]))
        steps2_exp_ca(w_v1, w_ca1, w_v2, w_ca2)
    elif (xp == '2stepsk'):
        w_v2, w_ca2 = list(map(int, sys.argv[2:4]))
        steps2_exp_k(w_v2, w_ca2)

    exit(0)
