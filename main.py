from HH_opt import HH_opt
from HH_simul import HH_simul
from Hodghux import HodgkinHuxley
import params
import sys



if __name__ == '__main__':

    v_fix = False
    xp = None
    w_v = 1
    w_ca = 0

    if(len(sys.argv)>3):
        xp = sys.argv[1]
        w_v = int(sys.argv[2])
        w_ca = int(sys.argv[3])

    name = 'Classic'
    opt = HH_opt(init_p=params.PARAMS_RAND, init_state=params.INIT_STATE)
    sim = HH_simul(init_p=params.DEFAULT, init_state=params.INIT_STATE, t=params.t_train, i_inj=params.i_inj_train)
    loop_func = HodgkinHuxley.integ_comp

    if(xp == 'ica'):
        v_fix = True
        name = 'Icafromv'
        opt = HH_opt(init_p=params.PARAMS_RAND, init_state=params.INIT_STATE_ica)
        sim = HH_simul(init_p=params.DEFAULT, init_state=params.INIT_STATE_ica, t=params.t, i_inj=params.v_inj)
        loop_func = HodgkinHuxley.ica_from_v

    elif(xp == 'notauca'):
        name = 'Notauca'
        loop_func = HodgkinHuxley.no_tau_ca

    elif(xp == 'notau'):
        name = 'Notau'
        loop_func = HodgkinHuxley.no_tau

        
    print(name, w_v, w_ca, loop_func)
    opt.loop_func = loop_func
    sim.loop_func = loop_func
    sim.Main(v_fix=v_fix, dump=True)
    opt.Main('%s_v=%s_ca=%s'%(name, w_v, w_ca), w=[w_v, w_ca])

    exit(0)
