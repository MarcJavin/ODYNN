from HH_opt import HH_opt
from HH_simul import HH_simul
from Hodghux import HodgkinHuxley
import params, data
import utils
import sys
import numpy as np
import scipy as sp


CA_VAR = {'e__tau', 'e__mdp', 'e__scale', 'f__tau', 'f__mdp', 'f__scale', 'h__alpha', 'h__mdp', 'h__scale', 'g_Ca', 'E_Ca', 'rho_ca', 'decay_ca'}
K_VAR = {'p__tau', 'p__mdp', 'p__scale', 'q__tau', 'q__mdp', 'q__scale', 'n_tau', 'n__mdp', 'n__scale', 'g_Kf', 'g_Ks', 'E_K'}

CA_CONST = params.ALL - CA_VAR
K_CONST = params.ALL - K_VAR

pars = [params.give_rand() for i in range(10)]

"""Single optimisation"""
def single_exp(xp, w_v, w_ca, suffix=None):
    name = 'Classic'

    opt = HH_opt(init_p=params.give_rand())
    sim = HH_simul(init_p=params.DEFAULT, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp

    if (xp == 'ica'):
        name = 'Icafromv'
        opt = HH_opt(init_p=params.give_rand(), fixed=CA_CONST)
        sim = HH_simul(init_p=params.DEFAULT, t=params.t, i_inj=params.v_inj)
        loop_func = HodgkinHuxley.ica_from_v

    elif(xp == 'ik'):
        name = 'Ikfromv'
        opt = HH_opt(init_p=params.give_rand(), fixed=K_CONST)
        sim = HH_simul(init_p=params.DEFAULT, t=params.t, i_inj=params.v_inj_rev)
        loop_func = HodgkinHuxley.ik_from_v

    elif (xp == 'notauca'):
        name = 'Notauca'
        loop_func = HodgkinHuxley.no_tau_ca

    elif (xp == 'notau'):
        name = 'Notau'
        loop_func = HodgkinHuxley.no_tau

    elif (xp == 'classic'):
        name = 'integcomp'
        loop_func = HodgkinHuxley.integ_comp

    print(name, w_v, w_ca, loop_func)
    dir = '%s_v=%s_ca=%s' % (name, w_v, w_ca)
    if (suffix is not None):
        dir = '%s_%s' % (dir, suffix)
    utils.set_dir(dir)
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    file = sim.simul(show=True, dump=True)
    opt.optimize(dir, w=[w_v, w_ca], file=file)
    return dir


def steps2_exp_ca(w_v1, w_ca1, w_v2, w_ca2):
    name = '_2steps'

    dir = single_exp('ica', w_v1, w_ca1, suffix='%s%s%s' % (name, w_v2, w_ca2))

    param = utils.get_dic_from_var(dir)
    opt = HH_opt(init_p=param, fixed=CA_VAR, l_rate=[0.1,9,0.9])
    sim = HH_simul(init_p=params.DEFAULT, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    file = sim.simul(dump=True, suffix='step2', show=False)
    opt.optimize(dir, w=[w_v2, w_ca2], suffix='step2', file=file)

    test_xp(dir)

def steps2_exp_k(w_v2, w_ca2):
    name = '_2steps'

    dir = single_exp('ik', 1, 0, suffix='%s%s%s' % (name, w_v2, w_ca2))

    param = utils.get_dic_from_var(dir)
    opt = HH_opt(init_p=param, fixed=K_VAR, l_rate=[0.1,9,0.9])
    sim = HH_simul(init_p=params.DEFAULT, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    file = sim.simul(dump=True, suffix='step2')
    opt.optimize(dir, w=[w_v2, w_ca2], suffix='step2', file=file)

    test_xp(dir)



def test_xp(dir, suffix='', show=False):

    dt = 0.05
    t = np.array(sp.arange(0.0, 4000, dt))
    t3 = np.array(sp.arange(0.0, 6000, dt))
    i1 = (t-1000)*(30./200)*((t>1000)&(t<=1200)) + 30*((t>1200)&(t<=3000)) - (t-2800)*(30./200)*((t>2800)&(t<=3000))
    i2 = (t - 1000) * (50. / 1000) * ((t > 1000) & (t <= 2000)) + (3000 - t) * (50. / 1000) * ((t > 2000) & (t <= 3000))
    i3 = (t3-1000)*(1./2000)*((t3>1000)&(t3<=3000)) + (5000-t3)*(1./2000)*((t3>3000)&(t3<=5000))

    utils.set_dir(dir)
    param = data.get_vars(dir)
    sim = HH_simul(init_p=param, t=t, i_inj=i1)
    sim.simul(show=show, suffix='xp1')
    sim.i_inj = i2
    sim.simul(show=show, suffix='xp2')
    sim.i_inj=i3
    sim.t = t3
    sim.simul(show=show, suffix='xp3')

def alternate(name=''):
    loop_func = HodgkinHuxley.integ_comp
    opt = HH_opt(init_p=pars, loop_func=loop_func)
    sim = HH_simul(init_p=params.DEFAULT, t=params.t_train, i_inj=params.i_inj_train, loop_func=loop_func)
    sim.simul(show=False, dump=True)
    dir = 'Integcomp_alternate_%s' % name
    wv = 0.2
    wca = 0.8
    opt.optimize(dir, [wv, wca], epochs=20, step=0)
    for i in range(10):
        wv = 1 - wv
        wca = 1 - wca
        opt.optimize(dir, [wv, wca], reload=True, epochs=20, step=i + 1)
    comp_pars(dir)
    test_xp(dir)


def only_calc(name=''):
    dt=0.1
    loop_func = HodgkinHuxley.integ_comp
    opt = HH_opt(init_p=pars, loop_func=loop_func, dt=dt)
    t,i = params.give_train(dt)
    sim = HH_simul(init_p=params.DEFAULT, t=t, i_inj=i, loop_func=loop_func)
    sim.simul(show=False, dump=True)
    dir = 'Integcomp_calc_%s'%name
    wv = 0
    wca = 1
    opt.optimize(dir, [wv, wca], epochs=240)
    comp_pars(dir)
    test_xp(dir)

def comp_pars(dir):
    p = data.get_vars(dir)
    utils.set_dir(dir)
    utils.plot_vars(p, func=utils.bar, suffix='compared', show=False, save=True)


if __name__ == '__main__':

    # comp_pars('server/Integcomp_alternate_10')


    xp = sys.argv[1]
    if(xp == 'alt'):
        name = sys.argv[2]
        alternate(name)
    elif(xp=='cac'):
        name = sys.argv[2]
        only_calc(name)
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
