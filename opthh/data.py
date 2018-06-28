"""
.. module:: data
    :synopsis: Module for data extraction and preparation

.. moduleauthor:: Marc Javin
"""

import math
import pickle
import time

import numpy as np
import pandas as pd
import pylab as plt
import scipy as sp
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter

from opthh import utils

DUMP_FILE = 'data/dump'
DUMP_real = 'data/real'
DUMP_real_all = 'data/real_all'
SAVE_PATH = 'tmp/model.ckpt'
FILE_LV = 'tmp/dump_lossratevars'
FILE_NEUR = 'tmp/neuron'
FILE_CIRC = 'tmp/circuit'
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

def get_vars(dir, i=-1):
    """get dic of vars from dumped file"""
    file = utils.RES_DIR + dir + '/' + FILE_LV
    with open(file, 'rb') as f:
        l,r,dic = pickle.load(f)
        dic = dict([(var, np.array(val[i], dtype=np.float32)) for var, val in dic.items()])
    return dic

def get_vars_all(dir, i=-1):
    """get dic of vars from dumped file"""
    file = utils.RES_DIR + dir + '/' + FILE_LV
    with open(file, 'rb') as f:
        l,r,dic = pickle.load(f)
        dic = dict([(var, val[:i]) for var, val in dic.items()])
    return dic

def get_best_result(dir, i=-1):
    file = utils.RES_DIR + dir + '/' + FILE_LV
    with open(file, 'rb') as f:
        l, r, dic = pickle.load(f)
        idx = np.nanargmin(l[-1])
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

def check_alpha(tinit, i, trace):
    """study the hill equation"""
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

def dump_data(delta=500, final_time=4000., dt=0.2):
    """dump real data into our format"""
    df = pd.read_csv('data/AVAL1.csv')
    # df = df.head(510)
    trace = np.array(df['trace'])*10

    unit_time = final_time/delta
    t = sp.arange(0., final_time, dt)
    i = np.array(df['inputCurrent']) * 10
    intervals = [0, 420, 1140, 2400]
    curs = np.zeros((len(t), len(intervals)))
    cas = np.zeros(curs.shape)
    for j, st in enumerate(intervals):
        td = np.arange(0., final_time, unit_time)
        ca = trace[st:st + delta]
        spl = splrep(td, ca, s=0.25)
        s_ca = splev(t, spl)
        spli = splrep(td, i[st:st+delta])
        s_i = splev(t, spli)
        plt.subplot(len(intervals), 1, j + 1)
        plt.plot(td, ca)
        plt.plot(t, s_ca)
        curs[:,j] = s_i
        cas[:,j] = s_ca
    plt.show()
    plt.close()
    with open(DUMP_real, 'wb') as f:
        pickle.dump([t, curs, None, cas], f)
    t_all = sp.arange(0., len(trace)*unit_time, dt)
    td_all = sp.arange(0., len(trace))*unit_time
    spl = splrep(td_all, trace, s=0.25)
    s_ca_all = splev(t_all, spl)
    spli = splrep(td_all, i)
    s_i_all = splev(t_all, spli)
    plt.plot(td_all, trace)
    plt.plot(t_all, s_ca_all)
    plt.show()
    with open(DUMP_real_all, 'wb') as f:
        pickle.dump([t_all, s_i_all, None, s_ca_all], f)
    return DUMP_real, DUMP_real_all




if __name__ == '__main__':
    df = pd.read_csv('data/SMDBoxes.csv')
    df = df.head(1510)
    trace = np.array(df['Calcium_bc'])
    i = np.array(df['Input']) * 100
    tinit = np.array(df['time [s]']) * 1000
    t = np.arange(0, tinit[-1], step=2)


    t1 = time.time()

    # f = interp1d(tinit, l, kind='cubic')
    # z = f(t)
    t2 = time.time()
    print('lowess+interp : %s'%(t2-t1))

    t1 = time.time()
    exact = splrep(tinit, trace, k=1)
    spl = splrep(tinit, trace, s=0.25)
    zexact = splev(tinit, exact)
    z = splev(tinit, spl)
    t2 = time.time()
    print('splrep : %s' % (t2-t1))

    spli = splrep(tinit, i, k=2)
    i = splev(tinit, spli)

    plt.subplot(211)
    plt.plot(tinit, trace, 'r', label='trace')
    plt.plot(tinit, z, 'b', label='splrev')
    plt.legend()
    plt.subplot(212)
    plt.plot(i)
    plt.show()

    pickle.dump([t, i, z], open(DUMP_FILE, 'wb'))
