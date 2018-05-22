from HH_opt import HH_opt
from HH_simul import HH_simul
from Hodghux import HodgkinHuxley
import params
import utils
import sys



"""Single optimisation"""
def single_exp(xp, w_v, w_ca, sufix=''):
    v_fix = False
    name = 'Classic'
    opt = HH_opt(init_p=params.PARAMS_RAND, init_state=params.INIT_STATE)
    sim = HH_simul(init_p=params.DEFAULT, init_state=params.INIT_STATE, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp

    if (xp == 'ica'):
        v_fix = True
        name = 'Icafromv'
        opt = HH_opt(init_p=params.PARAMS_RAND, init_state=params.INIT_STATE_ica)
        sim = HH_simul(init_p=params.DEFAULT, init_state=params.INIT_STATE_ica, t=params.t, i_inj=params.v_inj)
        loop_func = HodgkinHuxley.ica_from_v

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
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    sim.Main(v_fix=v_fix, dump=True)
    dir = '%s_v=%s_ca=%s'%(name+sufix, w_v, w_ca)
    opt.Main(dir, w=[w_v, w_ca])
    return dir


if __name__ == '__main__':

    name = '_2steps'
    w_v = 1
    w_ca = 0

    if(len(sys.argv) > 2):
        w_v, w_ca = sys.argv[1:3]

    dir = single_exp('ica', 1, 0, name)

    params = utils.get_dic_from_var(dir)
    opt = HH_opt(init_p=params.params, init_state=params.INIT_STATE)
    sim = HH_simul(init_p=params.DEFAULT, init_state=params.INIT_STATE, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    sim.Main(dump=True)
    dir = '%s_v=%s_ca=%s'
    opt.Main(dir, w=[w_v, w_ca], sufix='step2')


    exit(0)
