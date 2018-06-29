

import utils, datas
from neuronsimul import NeuronSimul


if __name__ == '__main__':
    t, i = datas.give_train(dt=1)
    dir = "hola"
    utils.set_dir(dir)
    sim = NeuronSimul(t=t, i_inj=i)
    train = sim.simul(show=True, suffix='train')
    T, I, V, Ca = train
    print(I.shape, V.shape)