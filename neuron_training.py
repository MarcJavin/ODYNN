import sys

import numpy as np
import scipy as sp

import data
import neuron_params
import params
import utils
from Neuron import HodgkinHuxley, NeuronLSTM
from NeuronOpt import NeuronOpt
from NeuronSimul import NeuronSimul

CA_VAR = {'e__tau', 'e__mdp', 'e__scale', 'f__tau', 'f__mdp', 'f__scale', 'h__alpha', 'h__mdp', 'h__scale', 'g_Ca', 'E_Ca', 'rho_ca', 'decay_ca'}
K_VAR = {'p__tau', 'p__mdp', 'p__scale', 'q__tau', 'q__mdp', 'q__scale', 'n_tau', 'n__mdp', 'n__scale', 'g_Kf', 'g_Ks', 'E_K'}

CA_CONST = neuron_params.ALL - CA_VAR
K_CONST = neuron_params.ALL - K_VAR

pars = [neuron_params.give_rand() for i in range(100)]
pars = data.get_vars('Init_settings_100_2', 0)
pars = [dict([(k, v[n]) for k, v in pars.items()]) for n in range(len(pars['C_m']))]
dt=1
t,i_inj = params.give_train(dt)
# i_inj = i_inj[:,2][:,None]



"""Single optimisation"""
def single_exp(xp, w_v, w_ca, suffix=None):
    name = 'Classic'

    opt = NeuronOpt(init_p=neuron_params.give_rand())
    sim = NeuronSimul(init_p=neuron_params.DEFAULT, t=t, i_inj=i_inj)
    base = HodgkinHuxley.step_model

    if (xp == 'ica'):
        name = 'Icafromv'
        opt = NeuronOpt(init_p=neuron_params.give_rand(), fixed=CA_CONST)
        sim = NeuronSimul(init_p=neuron_params.DEFAULT, t=t, i_inj=i_inj)
        loop_func = HodgkinHuxley.ica_from_v

    elif(xp == 'ik'):
        name = 'Ikfromv'
        opt = NeuronOpt(init_p=neuron_params.give_rand(), fixed=K_CONST)
        sim = NeuronSimul(init_p=neuron_params.DEFAULT, t=t, i_inj=i_inj)
        loop_func = HodgkinHuxley.ik_from_v

    elif (xp == 'notauca'):
        name = 'Notauca'
        loop_func = HodgkinHuxley.no_tau_ca

    elif (xp == 'notau'):
        name = 'Notau'
        loop_func = HodgkinHuxley.no_tau

    elif (xp == 'classic'):
        name = 'integcomp'
        loop_func = base

    print(name, w_v, w_ca, loop_func)
    dir = '%s_v=%s_ca=%s' % (name, w_v, w_ca)
    if (suffix is not None):
        dir = '%s_%s' % (dir, suffix)
    utils.set_dir(dir)
    HodgkinHuxley.step_model = loop_func
    HodgkinHuxley.step_model = loop_func
    file = sim.simul(show=True, dump=True)
    opt.optimize(dir, w=[w_v, w_ca], file=file)
    HodgkinHuxley.step_model = base
    return dir


def steps2_exp_ca(w_v1, w_ca1, w_v2, w_ca2):
    name = '_2steps'

    dir = single_exp('ica', w_v1, w_ca1, suffix='%s%s%s' % (name, w_v2, w_ca2))

    param = utils.get_dic_from_var(dir)
    opt = NeuronOpt(init_p=param, fixed=CA_VAR)
    sim = NeuronSimul(init_p=neuron_params.DEFAULT, t=t, i_inj=i_inj)
    file = sim.simul(dump=True, suffix='step2', show=False)
    opt.optimize(dir, w=[w_v2, w_ca2], l_rate=[0.1,9,0.9],suffix='step2', file=file)

    test_xp(dir)

def steps2_exp_k(w_v2, w_ca2):
    name = '_2steps'

    dir = single_exp('ik', 1, 0, suffix='%s%s%s' % (name, w_v2, w_ca2))

    param = utils.get_dic_from_var(dir)
    opt = NeuronOpt(init_p=param, fixed=K_VAR)
    sim = NeuronSimul(init_p=neuron_params.DEFAULT, t=t, i_inj=i_inj)
    file = sim.simul(dump=True, suffix='step2')
    opt.optimize(dir, w=[w_v2, w_ca2], l_rate=[0.1,9,0.9], suffix='step2', file=file)

    test_xp(dir)



def test_xp(dir, i=i_inj, default=neuron_params.DEFAULT, suffix='', show=False):

    utils.set_dir(dir)
    param = data.get_best_result(dir)
    for j, i_ in enumerate(i.transpose()):
        sim = NeuronSimul(init_p=param, t=t, i_inj=i_)
        sim.comp_targ(param, default, show=show, save=True, suffix='train%s'%j)

    dt = 0.05
    tt = np.array(sp.arange(0.0, 4000, dt))
    t3 = np.array(sp.arange(0.0, 6000, dt))
    i1 = (tt-1000)*(30./200)*((tt>1000)&(tt<=1200)) + 30*((tt>1200)&(tt<=3000)) - (tt-2800)*(30./200)*((tt>2800)&(tt<=3000))
    i2 = (tt - 1000) * (50. / 1000) * ((tt > 1000) & (tt <= 2000)) + (3000 - tt) * (50. / 1000) * ((tt > 2000) & (tt <= 3000))
    i3 = (t3-1000)*(1./2000)*((t3>1000)&(t3<=3000)) + (5000-t3)*(1./2000)*((t3>3000)&(t3<=5000))
    is_ = [i1,i2,i3]
    ts_ = [tt,tt,t3]
    for j, i_ in enumerate(is_):
        sim = NeuronSimul(init_p=param, t=ts_[j], i_inj=i_)
        sim.comp_targ(param, default, show=show, save=True, suffix='test%s' % j)

def alternate(name=''):
    dir = 'Integcomp_alternate_%s' % name
    utils.set_dir(dir)
    sim = NeuronSimul(init_p=neuron_params.DEFAULT, t=t, i_inj=i_inj)
    file = sim.simul(show=False, suffix='train', dump=True)
    wv = 0.2
    wca = 0.8
    opt = NeuronOpt(init_p=pars, dt=dt)
    opt.optimize(dir, [wv, wca], epochs=20, step=0, file=file)
    for i in range(24):
        wv = 1 - wv
        wca = 1 - wca
        n = opt.optimize(dir, [wv, wca], reload=True, epochs=20, step=i + 1, file=file)
    comp_pars(dir, n)
    test_xp(dir)


def classic(name, wv, wca, default=neuron_params.DEFAULT_2):
    if(wv == 0):
        dir = 'Integcomp_calc_%s' % name
    elif(wca == 0):
        dir = 'Integcomp_volt_%s' % name
    else:
        dir = 'Integcomp_both_%s' % name
    utils.set_dir(dir)
    neur = NeuronLSTM(dt=dt)
    opt = NeuronOpt(neur)
    sim = NeuronSimul(init_p=default, t=t, i_inj=i_inj)
    file = sim.simul(show=False, suffix='train', dump=True)
    n = opt.optimize(dir, w=[wv, wca], epochs=500, file=file)
    comp_pars(dir, n)
    test_xp(dir, default=default)

def real_data(name):
    dir = 'Real_data_%s' % name
    utils.set_dir(dir)
    opt = NeuronOpt(init_p=pars, dt=dt)
    filetrain, filetest = data.dump_data()
    n = opt.optimize(dir, w=[0, 1], epochs=500, file=filetrain)
    comp_pars(dir, n)
    t,i,v,ca = data.get_data_dump(filetest)
    sim = NeuronSimul(init_p=data.get_best_result(dir), t=t, i_inj=i)
    sim.simul(suffix='test', save=True, ca_true=ca)


def comp_pars(dir, i=-1):
    p = data.get_vars(dir, i)
    utils.set_dir(dir)
    utils.plot_vars(p, func=utils.bar, suffix='compared', show=False, save=True)
    utils.boxplot_vars(p, suffix='boxes', show=False, save=True)


def add_plots():
    import glob
    import re
    for filename in glob.iglob(utils.RES_DIR+'*'):
        dir = re.sub(utils.RES_DIR, '', filename)
        try:
            comp_pars(dir)
        except:
            print(dir)


if __name__ == '__main__':

    

    xp = sys.argv[1]
    if(xp == 'alt'):
        name = sys.argv[2]
        alternate(name)
    elif(xp=='cac'):
        name = sys.argv[2]
        classic(name, wv=0, wca=1)
    elif(xp=='v'):
        name = sys.argv[2]
        classic(name, wv=1, wca=0)
    elif (xp == 'both'):
        name = sys.argv[2]
        classic(name, wv=1, wca=1)
    elif(xp=='real'):
        name = sys.argv[2]
        real_data(name)
    elif(xp == 'single'):
        xp = sys.argv[2]
        w_v, w_ca = list(map(int, sys.argv[3:5]))
        single_exp(xp, w_v, w_ca)
    elif(xp == '2stepsca'):
        w_v1, w_ca1, w_v2, w_ca2 = list(map(int, sys.argv[2:6]))
        steps2_exp_ca(w_v1, w_ca1, w_v2, w_ca2)
    elif (xp == '2stepsk'):
        w_v2, w_ca2 = list(map(int, sys.argv[2:4]))
        steps2_exp_k(w_v2, w_ca2)



    exit(0)
