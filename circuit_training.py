
from Circuit import Circuit
from HH_opt import HH_opt
import numpy as np
import scipy as sp
import params

connections = [(0,1), (1,0)]

t_train = np.array(sp.arange(0.0, 1200., params.DT))
i0 = 10.*((t_train>100)&(t_train<400))
i1 = 10.*((t_train>600)&(t_train<900))
i_out = 10.*((t_train>350)&(t_train<700))

i_injs = np.array([i0,
          i1])

neurons = [HH_opt(init_p=params.DEFAULT, fixed=params.ALL),
           HH_opt(init_p=params.DEFAULT, fixed=params.ALL)]






if __name__ == '__main__':
    c = Circuit(neurons, connections, i_injs, t_train, i_out, init_p=params.SYNAPSE)
    c.run_sim()