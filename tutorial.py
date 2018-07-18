

if __name__ == '__main__':
    import opthh.nsimul as sim
    import scipy as sp

    t = sp.arange(0., 1200., 0.1)
    i = 40. * ((t > 400) & (t < 800))
    sim.simul(dt=0.1, i_inj=i, show=True, save=False)

    from opthh.hhmodel import DEFAULT, DEFAULT_2

    sim.comp_pars_targ(DEFAULT, DEFAULT_2, show=True)