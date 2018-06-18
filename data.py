# from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d, splrep, splev
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import pylab as plt
import time
import pickle
import math
import utils

DUMP_FILE = 'data/dump'
SAVE_PATH = 'tmp/model.ckpt'
FILE_LV = 'tmp/dump_lossratevars'
FILE_NEUR = 'tmp/neuron'
FILE_CIRC = 'tmp/circuit'
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

"""get dic of vars from dumped file"""
def get_vars(dir, i=-1):
    file = utils.RES_DIR+dir+'/'+FILE_LV
    with open(file, 'rb') as f:
        l,r,dic = pickle.load(f)
        dic = dict([(var, np.array(val[i], dfype=np.float32)) for var, val in dic.items()])
    return dic

"""get dic of vars from dumped file"""
def get_vars_all(dir, i=-1):
    file = utils.RES_DIR+dir+'/'+FILE_LV
    with open(file, 'rb') as f:
        l,r,dic = pickle.load(f)
        dic = dict([(var, val[:i]) for var, val in dic.items()])
    return dic

def get_best_result(dir, i=-1):
    file = utils.RES_DIR + dir + '/' + FILE_LV
    with open(file, 'rb') as f:
        l, r, dic = pickle.load(f)
        idx = np.argmin.min(l[-1])
        dic = dict([(var, val[i,idx]) for var, val in dic.items()])
    return dic

def get_data_dump(file=DUMP_FILE):
    with open(file, 'rb') as f:
        T, X, V, Ca = pickle.load(f)
    return T, X, V, Ca

def get_data_dump2(file=DUMP_FILE):
    with open(file, 'rb') as f:
        T, X, Ca = pickle.load(f)
    return T, X, Ca


def get_data(file='AVAL_test.csv'):
    df = pd.read_csv(file)
    Y = np.array(df['trace'])
    X = np.array(df['inputCurrent']) * 10 + np.full(Y.shape, 0.001)
    T = np.array(df['timeVector']) * 1000
    return T, X, Y

"""study the hill equation"""
def check_alpha(tinit, i, trace):
    vals = np.logspace(math.log10(0.1), math.log10(100.), num=20)
    idx=0
    plt.subplot(211)
    plt.plot(trace)
    spl = splrep(tinit, trace, s=0.5)
    trace = splev(tinit, spl)
    plt.plot(trace)
    plt.subplot(212)
    plt.plot(i)
    plt.show()
    for alpha in vals:
        idx += 1
        k = 189.e-6
        n = 3.8
        bas = (-k*trace) / (trace - np.full(trace.shape, alpha))
        cac = np.power(bas, n)
        plt.subplot(len(vals)/4, 4, idx)
        plt.plot(cac, label='alpha=%.2f'%alpha)
        z2 = savgol_filter(cac, 9, 3)
        plt.plot(z2, 'r', label='smooth')
        plt.legend()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('data/AVAL1.csv')
    df = df.head(510)
    trace = np.array(df['trace'])
    i = np.array(df['inputCurrent'])*10
    tinit = np.array(df['timeVector'])*1000
    t = np.arange(0,tinit[-1],step=2)
    print(trace.shape)


    t1 = time.time()

    # f = interp1d(tinit, l, kind='cubic')
    # z = f(t)
    t2 = time.time()
    print('lowess+interp : %s'%(t2-t1))

    t1 = time.time()
    exact = splrep(tinit, trace, k=1)
    spl = splrep(tinit, trace, s=0.25)
    zexact = splev(t, exact)
    z = splev(t, spl)
    t2 = time.time()
    print('splrep : %s' % (t2-t1))

    spli = splrep(tinit, i, k=2)
    i = splev(t, spli)

    plt.subplot(211)
    plt.plot(tinit, trace, 'r', label='trace')
    plt.plot(t, z, 'b', label='splrev')
    plt.legend()
    plt.subplot(212)
    plt.plot(t, i)
    plt.show()

    pickle.dump([t, i, z], open(DUMP_FILE, 'wb'))
