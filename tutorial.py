

if __name__ == '__main__':
    from opthh.neuronsimul import NeuronSimul
    import scipy as sp

    t = sp.arange(0., 1200., 0.1)
    i = 40. * ((t > 400) & (t < 800))
    simul = NeuronSimul(t=t, i_inj=i)
    simul.simul(show=True)

    from opthh.hhmodel import DEFAULT, DEFAULT_2

    simul.comp_pars_targ(DEFAULT, DEFAULT_2, show=True)