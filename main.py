from HH_opt import HH_opt
from HH_simul import HH_simul
from Hodghux import HodgkinHuxley
import params
import utils
import sys
import numpy as np
import scipy as sp


CA_VAR = ['e__tau', 'e__mdp', 'e__scale', 'f__tau', 'f__mdp', 'f__scale', 'h__alpha', 'h__mdp', 'h__scale', 'g_Ca', 'E_Ca']
K_VAR = ['p__tau', 'p__mdp', 'p__scale', 'q__tau', 'q__mdp', 'q__scale', 'n_tau', 'n__mdp', 'n__scale', 'g_Kf', 'g_Ks', 'E_K']
CA_CONST = []
K_CONST = []
for k in params.DEFAULT.keys():
    if k not in CA_VAR:
        CA_CONST.append(k)
    if k not in K_VAR:
        K_CONST.append(k)


"""Single optimisation"""
def single_exp(xp, w_v, w_ca, sufix=None):
    v_fix = False
    name = 'Classic'

    opt = HH_opt(init_p=params.PARAMS_RAND)
    sim = HH_simul(init_p=params.DEFAULT, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp

    if (xp == 'ica'):
        name = 'Icafromv'
        opt = HH_opt(init_p=params.PARAMS_RAND, fixed=CA_CONST, epochs=180)
        sim = HH_simul(init_p=params.DEFAULT, t=params.t, i_inj=params.v_inj)
        loop_func = HodgkinHuxley.ica_from_v

    elif(xp == 'ik'):
        name = 'Ikfromv'
        opt = HH_opt(init_p=params.PARAMS_RAND, fixed=K_CONST, epochs=180)
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
    if (sufix is not None):
        dir = '%s_%s' % (dir, sufix)
    utils.set_dir(dir)
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    sim.Main(show=True, dump=True)
    opt.Main(dir, w=[w_v, w_ca])
    return dir


def steps2_exp_ca(w_v1, w_ca1, w_v2, w_ca2):
    name = '_2steps'

    dir = single_exp('ica', w_v1, w_ca1, sufix='%s%s%s' % (name, w_v2, w_ca2))

    param = utils.get_dic_from_var(dir)
    opt = HH_opt(init_p=param, fixed=CA_VAR, l_rate=[0.1,9,0.9])
    sim = HH_simul(init_p=params.DEFAULT, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    sim.Main(dump=True, sufix='step2')
    opt.Main(dir, w=[w_v2, w_ca2], sufix='step2')

    test_xp(dir)

def steps2_exp_k(w_v2, w_ca2):
    name = '_2steps'

    dir = single_exp('ik', 1, 0, sufix='%s%s%s' % (name, w_v2, w_ca2))

    param = utils.get_dic_from_var(dir)
    opt = HH_opt(init_p=param, fixed=K_VAR, l_rate=[0.1,9,0.9])
    sim = HH_simul(init_p=params.DEFAULT, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    sim.Main(dump=True, sufix='step2')
    opt.Main(dir, w=[w_v2, w_ca2], sufix='step2')

    test_xp(dir)



def test_xp(dir, show=False):

    dt = params.DT
    t = np.array(sp.arange(0.0, 4000, dt))
    t3 = np.array(sp.arange(0.0, 6000, dt))
    i1 = (t-1000)*(30./200)*((t>1000)&(t<=1200)) + 30*((t>1200)&(t<=3000)) - (t-2800)*(30./200)*((t>2800)&(t<=3000))
    i2 = (t - 1000) * (50. / 1000) * ((t > 1000) & (t <= 2000)) + (3000 - t) * (50. / 1000) * ((t > 2000) & (t <= 3000))
    i3 = (t3-1000)*(1./2000)*((t3>1000)&(t3<=3000)) + (5000-t3)*(1./2000)*((t3>3000)&(t3<=5000))

    utils.set_dir(dir)
    param = utils.get_dic_from_var(dir)
    sim = HH_simul(init_p=param, t=t, i_inj=i1)
    sim.Main(show=show, sufix='xp1')
    sim.i_inj = i2
    sim.Main(show=show, sufix='xp2')
    sim.i_inj=i3
    sim.t = t3
    sim.Main(show=show, sufix='xp3')


if __name__ == '__main__':

    opt = HH_opt(init_p=params.PARAMS_RAND, epochs=50)
    sim = HH_simul(init_p=params.DEFAULT, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    sim.Main(show=True, dump=True)
    dir = 'Integcomp_alternate'
    wv = 0
    wca = 1
    opt.Main(dir, [wv, wca], 'step0')
    for i in range(5):
        new_params = utils.get_dic_from_var(dir, 'step%s' % i)
        opt.change_params(new_params)
        wv = 1-wv
        wca = 1-wca
        opt.Main(dir, [wv, wca], 'step%s'%(i+1))


    exit(0)


    xp = sys.argv[1]
    if(xp == 'single'):
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
