from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d, splrep, splev
import numpy as np
import pandas as pd
import pylab as plt
import time
import pickle

DUMP_FILE = 'data.txt'


def get_data_dump(file=DUMP_FILE):
    with open(file, 'rb') as f:
        T, X, V, Ca = pickle.load(f)
    return T, X, V, Ca

if __name__ == '__main__':
    dt = pd.read_csv('AVAL1.csv')
    dt = dt.head(100)
    trace = np.array(dt['trace'])
    tinit = np.array(dt['timeVector'])
    t = np.arange(0,tinit[-1],step=0.001)

    t1 = time.time()
    l = lowess(trace, tinit, return_sorted=False, frac=0.01)

    f = interp1d(tinit, l, kind='cubic')
    z = f(t)
    t2 = time.time()
    print('lowess+interp : %s'%(t2-t1))

    t1 = time.time()
    exact = splrep(tinit, trace, k=1)
    spl = splrep(tinit, trace, s=0.05)
    zexact = splev(t, exact)
    z2 = splev(t, spl)
    t2 = time.time()
    print('splrep : %s' % (t2-t1))

    # plt.subplot(211)
    # plt.plot(trace)
    # plt.plot(l)
    # plt.subplot(212)
    plt.plot(z, 'g', label='lowess+interp1d')
    plt.plot(z2, 'b', label='splrev')
    plt.plot(zexact, 'r', label='exact')
    plt.legend()
    plt.show()

    pickle.dump()
