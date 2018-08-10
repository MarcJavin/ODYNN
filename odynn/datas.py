"""
.. module:: data
    :synopsis: Module for data extraction and preparation

.. moduleauthor:: Marc Javin
"""

import math
import numpy as np
import pandas as pd
import pylab as plt
import scipy as sp
from scipy import signal
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter
from random import random as rd

DUMP_FILE= 'data'
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

try:
    df = pd.read_csv('data/AVAL1.csv').head(3100)
except:
    df = pd.read_csv('../data/AVAL1.csv').head(3100)

def check_alpha(show=True):
    """study the hill equation

    """
    d = df.head(1000)
    t = d['timeVector']
    i = d['inputCurrent']
    trace = d['trace']
    vals = np.logspace(math.log10(1.2), math.log10(20.), num=5)
    idx=1
    plt.subplot(6,1,1)
    plt.plot(trace, 'r')
    for alpha in vals:
        idx += 1
        k = 189.e-6
        n = 3.8
        bas = (-k*trace) / (trace - np.full(trace.shape, alpha))
        bas[bas < 0] = 0
        cac = np.power(bas, 1/n)
        plt.subplot(6, 1, idx)
        plt.plot(cac, label='$\\alpha$=%.2f'%alpha)
        # z2 = savgol_filter(cac, 9, 3)
        # plt.plot(z2, 'r', label='smooth')
        plt.legend()
    if (show):
        plt.show()
        plt.close()

def get_real_data_norm(file='data/AVAL{}.csv'):
    df = pd.read_csv(file.format(1)).head(3100)
    t = np.array(df['timeVector']) * 1000
    trace = np.array(df['trace'])
    i = np.array(df['inputCurrent']) * 10
    k = 189.e-6
    n = 3.8
    alpha = 2.
    bas = (-k * trace) / (trace - np.full(trace.shape, alpha))
    # bas[bas < 0] = 0
    plt.subplot(3,1,1)
    plt.plot(t, trace)
    plt.title('Calcium imaging data')
    cac = np.power(bas, 1 / n)
    plt.subplot(3,1,2)
    plt.plot(t, cac)
    plt.title('Calcium concentration')
    plt.subplot(3,1,3)
    plt.plot(t, i)
    plt.title('Input current')
    plt.show()
    df2 = pd.read_csv(file.format(2)).head(3100)
    trace = np.array(df2['trace']) * 10
    bas = (-k * trace) / (trace - np.full(trace.shape, alpha))
    bas[bas < 0] = 0
    cac2 = np.power(bas, 1 / n)
    t2 = np.array(df2['timeVector']) * 1000
    i2 = np.array(df2['inputCurrent']) * 10
    train = [t, i[:, np.newaxis], [None, cac[:, np.newaxis]]]
    test = [t, i2[:, np.newaxis], [None, cac2[:, np.newaxis]]]
    return train, test


def get_real_data(delta=500, final_time=4000., dt=0.2, show=False):
    """dump real data into our format

    Args:
      delta:  (Default value = 500)
      final_time:  (Default value = 4000.)
      dt(float): time step (Default value = 0.2)

    Returns:

    """
    # df = df.head(510)
    trace = np.array(df['trace'])*10

    unit_time = final_time/delta
    t = sp.arange(0., final_time, dt)
    i = np.array(df['inputCurrent']) * 10
    intervals = [0, 0, 420, 1140, 2400]
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
    if(show):
        plt.show()
    plt.close()
    train = [t, curs, [None, cas]]
    t_all = sp.arange(0., len(trace)*unit_time, dt)
    td_all = sp.arange(0., len(trace))*unit_time
    spl = splrep(td_all, trace, s=0.25)
    s_ca_all = splev(t_all, spl)
    spli = splrep(td_all, i)
    s_i_all = splev(t_all, spli)
    plt.plot(td_all, trace)
    plt.plot(t_all, s_ca_all)
    if (show):
        plt.show()
    plt.close()
    test = [t_all, s_i_all, [None, s_ca_all]]
    return train, test


DT = 0.1


def give_train(dt=DT, nb_neuron_zero=None, max_t=1200.):
    """time and currents for optimization

    Args:
      dt(float): time step (Default value = DT)
      nb_neuron_zero:  (Default value = None)
      max_t:  (Default value = 1200.)

    Returns:

    """
    t = np.array(sp.arange(0.0, max_t, dt))
    i = 10. * ((t > 100) & (t < 300)) + 20. * ((t > 400) & (t < 600)) + 40. * ((t > 800) & (t < 950))
    i2 = 30. * ((t > 100) & (t < 500)) + 25. * ((t > 800) & (t < 900))
    i3 = np.sum([(10. + (n * 2 / 100)) * ((t > n) & (t < n + 50)) for n in range(100, 1100, 100)], axis=0)
    i4 = 15. * ((t > 400) & (t < 800))
    i5 = (t - 450) * (8. / 550) * ((t > 100) & (t <= 1100))
    i_fin = np.stack([i, i2, i3, i4, i5], axis=1)
    i_noise = 0.1 * (np.random.rand(i_fin.shape[0], i_fin.shape[1]) - 0.5)
    i_fin += i_noise
    # plt.plot(i_noise[:,0])
    # plt.show()
    if nb_neuron_zero is not None:
        i_zeros = np.zeros(i_fin.shape)
        i_fin = np.stack([i_fin] + [i_zeros for _ in range(nb_neuron_zero)], axis=2)
    return t, i_fin


def give_periodic(t, max_i, size, freq):
    return np.sum([max_i * ((t > n) & (t < n + size)) for n in range(0, int(t[-1]), freq)], axis=0)


def give_train2(dt=DT):
    t = np.array(sp.arange(0.0, 1200., dt))
    b1 = 40. * t / 1200
    b2 = -b1/2
    b3 = 40. - b1
    b4 = -b3/2
    i_fin = np.stack([b1 * rd() + b2 * rd() + b3 * rd() + b4 * rd() + give_periodic(t, rd() * 15., rd()*200, int(rd() * 500)) +
             give_periodic(t, rd() * 15., rd()*200, int(rd() * 500)) + give_periodic(t, rd() * 15., rd()*200, int(rd() * 500))
                      for _ in range(10)], axis=1)
    return t, i_fin


def give_test(dt=DT, max_t=1200.):
    """time and currents for optimization

    Args:
      dt(float): time step (Default value = DT)

    Returns:

    """
    t = np.array(sp.arange(0.0, max_t, dt))
    i1 = (t - 100) * (30. / 100) * ((t > 100) & (t <= 200)) + 30 * ((t > 200) & (t <= 1100)) - (t - 1000) * (
            30. / 100) * ((t > 1000) & (t <= 1100))
    i2 = 30. * ((t > 100) & (t < 300)) + 15. * ((t > 400) & (t < 500)) + 10. * ((t > 700) & (t < 1000))
    i3 = (t - 600) * (1. / 500) * ((t > 100) & (t <= 600)) + (1100 - t) * (1. / 500) * (
            (t > 600) & (t <= 1100))
    i4 = np.sum([(30. - (n * 4 / 100)) * ((t > n) & (t < n + 50)) for n in range(100, 1100, 100)], axis=0)
    i5 = signal.gaussian(len(t), std=len(t)/5) * 20.
    i_fin = np.stack([i1, i2, i3, i4, i5], axis=1)
    return t, i_fin


def full4(dt=DT, nb_neuron_zero=None, max_t=1200.):
    t = np.array(sp.arange(0.0, max_t, dt))
    i1 = 10. * ((t > 200) & (t < 600))
    i2 = 10. * ((t > 300) & (t < 700))
    i3 = 10. * ((t > 400) & (t < 800))
    i4 = 10. * ((t > 500) & (t < 900))
    is_ = np.stack([i1, i2, i3, i4], axis=1)
    i1 = 20. * ((t > 400) & (t < 900))
    i2 = 10. * ((t > 300) & (t < 700))
    i3 = 30. * ((t > 200) & (t < 800))
    i4 = 10. * ((t > 600) & (t < 800))
    is_2 = np.stack([i1, i2, i3, i4], axis=1)
    i1 = np.sum([10. * ((t > n) & (t < n + 100)) for n in range(70, 1100, 200)], axis=0)
    i2 = np.sum([(10. + (n * 1 / 100)) * ((t > n) & (t < n + 50)) for n in range(100, 1100, 100)], axis=0)
    i3 = np.sum([(22. - (n * 1 / 100)) * ((t > n) & (t < n + 50)) for n in range(120, 1100, 100)], axis=0)
    i4 = np.sum([(10. + (n * 2 / 100)) * ((t > n) & (t < n + 20)) for n in range(100, 1100, 80)], axis=0)
    is_3 = np.stack([i1, i2, i3, i4], axis=1)
    i_fin = np.stack([is_, is_2, is_3], axis=1)
    if nb_neuron_zero is not None:
        i_zeros = np.zeros((len(t), i_fin.shape[1], nb_neuron_zero))
        i_fin = np.append(i_fin, i_zeros, axis=2)
    return t, i_fin

def full4_test(dt=DT, nb_neuron_zero=None, max_t=1200.):
    t = np.array(sp.arange(0.0, max_t, dt))
    i1 = 10. * ((t > 100) & (t < 200))
    i2 = 10. * ((t > 500) & (t < 700))
    i3 = 10. * ((t > 500) & (t < 800))
    i4 = 10. * ((t > 700) & (t < 900))
    is_ = np.stack([i1, i2, i3, i4], axis=1)
    i1 = 20. * ((t > 100) & (t < 1000))
    i2 = 10. * ((t > 900) & (t < 1000))
    i3 = 30. * ((t > 300) & (t < 700))
    i4 = 10. * ((t > 350) & (t < 950))
    is_2 = np.stack([i1, i2, i3, i4], axis=1)
    i1 = np.sum([5. * ((t > n) & (t < n + 100)) for n in range(70, 1100, 300)], axis=0)
    i2 = np.sum([(15. + (n * 3 / 100)) * ((t > n) & (t < n + 20)) for n in range(100, 1100, 50)], axis=0)
    i3 = np.sum([(12. - (n * 1 / 100)) * ((t > n) & (t < n + 50)) for n in range(120, 1100, 120)], axis=0)
    i4 = np.sum([(7. + (n * 0.5 / 100)) * ((t > n) & (t < n + 70)) for n in range(100, 1100, 80)], axis=0)
    is_3 = np.stack([i1, i2, i3, i4], axis=1)
    i_fin = np.stack([is_, is_2, is_3], axis=1)
    if nb_neuron_zero is not None:
        i_zeros = np.zeros((len(t), i_fin.shape[1], nb_neuron_zero))
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

if __name__ == '__main__':
    # get_real_data_norm()
    check_alpha()

    # give_train2(0.5)
    # exit(0)
    #
    # df = pd.read_csv('data/SMDBoxes.csv')
    # df = df.head(1510)
    # trace = np.array(df['Calcium_bc'])
    # i = np.array(df['Input']) * 100
    # tinit = np.array(df['time [s]']) * 1000
    # t = np.arange(0, tinit[-1], step=2)
    #
    #
    # t1 = time.time()
    #
    # # f = interp1d(tinit, l, kind='cubic')
    # # z = f(t)
    # t2 = time.time()
    # print('lowess+interp : %s'%(t2-t1))
    #
    # t1 = time.time()
    # exact = splrep(tinit, trace, k=1)
    # spl = splrep(tinit, trace, s=0.25)
    # zexact = splev(tinit, exact)
    # z = splev(tinit, spl)
    # t2 = time.time()
    # print('splrep : %s' % (t2-t1))
    #
    # spli = splrep(tinit, i, k=2)
    # i = splev(tinit, spli)
    #
    # plt.subplot(211)
    # plt.plot(tinit, trace, 'r', label='trace')
    # plt.plot(tinit, z, 'b', label='splrev')
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(i)
    # plt.show()



