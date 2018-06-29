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
    """Return parameters of the best optimized model"""
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
    train = [t, curs, None, cas]
    t_all = sp.arange(0., len(trace)*unit_time, dt)
    td_all = sp.arange(0., len(trace))*unit_time
    spl = splrep(td_all, trace, s=0.25)
    s_ca_all = splev(t_all, spl)
    spli = splrep(td_all, i)
    s_i_all = splev(t_all, spli)
    plt.plot(td_all, trace)
    plt.plot(t_all, s_ca_all)
    plt.show()
    test = [t_all, s_i_all, None, s_ca_all]
    return train, test




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

DT = 0.1


def give_train(dt=DT, nb_neuron_zero=None, max_t=1200.):
    """time and currents for optimization"""
    t = np.array(sp.arange(0.0, max_t, dt))
    i = 10. * ((t > 100) & (t < 300)) + 20. * ((t > 400) & (t < 600)) + 40. * ((t > 800) & (t < 950))
    i2 = 30. * ((t > 100) & (t < 500)) + 25. * ((t > 800) & (t < 900))
    i3 = np.sum([(10. + (n * 2 / 100)) * ((t > n) & (t < n + 50)) for n in range(100, 1100, 100)], axis=0)
    i4 = 15. * ((t > 400) & (t < 800))
    i_fin = np.stack([i, i2, i3, i4], axis=1)
    if nb_neuron_zero is not None:
        i_zeros = np.zeros(i_fin.shape)
        i_fin = np.stack([i_fin, i_zeros], axis=2)
    return t, i_fin


def give_test(dt=DT):
    """time and currents for optimization"""
    t = np.array(sp.arange(0.0, 1200, dt))
    i1 = (t - 100) * (30. / 100) * ((t > 100) & (t <= 200)) + 30 * ((t > 200) & (t <= 1100)) - (t - 1000) * (
            30. / 100) * ((t > 1000) & (t <= 1100))
    i2 = 30. * ((t > 100) & (t < 300)) + 15. * ((t > 400) & (t < 500)) + 10. * ((t > 700) & (t < 1000))
    i3 = (t - 600) * (1. / 500) * ((t > 100) & (t <= 600)) + (1100 - t) * (1. / 500) * (
            (t > 600) & (t <= 1100))
    i4 = np.sum([(30. - (n * 4 / 100)) * ((t > n) & (t < n + 50)) for n in range(100, 1100, 100)], axis=0)
    i_fin = np.stack([i1, i2, i3, i4], axis=1)
    return t, i_fin


def full4(dt=DT, nb_neuron_zero=None):
    t = np.array(sp.arange(0.0, 1200., dt))
    i1 = 10. * ((t > 200) & (t < 600))
    i2 = 10. * ((t > 300) & (t < 700))
    i3 = 10. * ((t > 400) & (t < 800))
    i4 = 10. * ((t > 500) & (t < 900))
    is_ = np.stack([i1, i2, i3, i4], axis=1)
    is_2 = is_ * 2
    i1 = np.sum([10. * ((t > n) & (t < n + 100)) for n in range(70, 1100, 200)], axis=0)
    i2 = np.sum([(10. + (n * 1 / 100)) * ((t > n) & (t < n + 50)) for n in range(100, 1100, 100)], axis=0)
    i3 = np.sum([(22. - (n * 1 / 100)) * ((t > n) & (t < n + 50)) for n in range(120, 1100, 100)], axis=0)
    i4 = np.sum([(10. + (n * 2 / 100)) * ((t > n) & (t < n + 20)) for n in range(100, 1100, 80)], axis=0)
    is_3 = np.stack([i1, i2, i3, i4], axis=1)
    i_fin = np.stack([is_, is_3], axis=1)
    if nb_neuron_zero is not None:
        i_zeros = np.zeros((len(t), i_fin.shape[1], 6))
        i_fin = np.append(i_fin, i_zeros, axis=2)
    return t, i_fin


def test():
    t, i = full4()
    print(i.shape)
    i_1 = np.zeros((len(t), i.shape[1], 6))
    print(i_1.shape)
    i = np.append(i, i_1, axis=2)
    print(i.shape)

t_len = 5000.
t = np.array(sp.arange(0.0, t_len, DT))
i_inj = 10. * ((t > 100) & (t < 750)) + 20. * ((t > 1500) & (t < 2500)) + 40. * ((t > 3000) & (t < 4000))
v_inj = 115 * (t / t_len) - np.full(t.shape, 65)
v_inj_rev = np.full(t.shape, 50) - v_inj
i_inj = np.array(i_inj, dtype=np.float32)

t_test = np.array(sp.arange(0.0, 2000, DT))
i_test = 10. * ((t_test > 100) & (t_test < 300)) + 20. * ((t_test > 400) & (t_test < 600)) + 40. * (
        (t_test > 800) & (t_test < 950)) + \
         (t_test - 1200) * (50. / 500) * ((t_test > 1200) & (t_test < 1700))
