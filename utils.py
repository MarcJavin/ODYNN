import socket
import matplotlib
if (socket.gethostname()=='1080'):
    matplotlib.use("Agg")
import pylab as plt
import numpy as np
import pandas as pd
import os
import pickle
import re

RES_DIR = 'results/'
IMG_DIR = 'img/'
DIR = RES_DIR
DUMP_FILE = 'data.txt'
OUT_PARAMS = 'params.txt'
OUT_SETTINGS = 'settings'
REGEX_VARS = '(.*) : (.*)'

RATE_COLORS = {'p' : '#00ccff',
               'q' : '#0000ff',
               'n' : '#cc00ff',
               'e' : '#b30000',
               'f' : '#ff9900',
               'h' : '#ff1a1a'
                }

def set_dir(subdir):
    global DIR
    DIR = RES_DIR + subdir
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(DIR+IMG_DIR):
        os.makedirs(DIR+IMG_DIR)
    return DIR

def get_data_dump(file=DUMP_FILE):
    with open(file, 'rb') as f:
        T, X, V, Ca = pickle.load(f)
    return T, X, V, Ca


"""Get variables values into a dictionnary"""
def get_dic_from_var(dir):
    file = RES_DIR + dir + '/' + OUT_PARAMS
    dic = {}
    with open(file, 'r') as f:
        for line in f:
            m = re.search(REGEX_VARS, line)
            dic[m.group(1)] = float(m.group(2))
    return dic




def get_data(file='AVAL_test.csv'):
    df = pd.read_csv(file)
    Y = np.array(df['trace'])
    X = np.array(df['inputCurrent']) * 10 + np.full(Y.shape, 0.001)
    T = np.array(df['timeVector']) * 1000
    return T, X, Y

"""plot variation of all variables organized by categories"""
def plot_vars(var_dic, lim, suffix="", show=True, save=False):
    for gate in ['e', 'f', 'n', 'p', 'q']:
        plot_vars_gate(gate, var_dic['%s__mdp'%gate][:lim+1], var_dic['%s__scale'%gate][:lim+1], var_dic['%s__tau'%gate][:lim+1], suffix, show, save)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(var_dic['h__mdp'][:lim+1], 'r')
    plt.ylabel('Midpoint')
    plt.title('h')
    plt.subplot(3, 1, 2)
    plt.plot(var_dic['h__scale'][:lim+1], 'g')
    plt.ylabel('Scale')
    plt.subplot(3, 1, 3)
    plt.plot(var_dic['h__alpha'][:lim+1])
    plt.ylabel('Time constant')
    if (show):
        plt.show()
    if (save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'h', suffix))

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(var_dic['g_Ks'][:lim + 1], RATE_COLORS['n'])
    plt.ylabel('KS conductance')
    plt.title('Conductances')
    plt.subplot(4, 1, 2)
    plt.plot(var_dic['g_Kf'][:lim + 1], RATE_COLORS['p'])
    plt.ylabel('KF conductance')
    plt.subplot(4, 1, 3)
    plt.plot(var_dic['g_Ca'][:lim + 1], RATE_COLORS['e'])
    plt.ylabel('Ca conductance')
    plt.subplot(4, 1, 4)
    plt.plot(var_dic['g_L'][:lim + 1], 'k')
    plt.ylabel('Leak conductance')
    if (show):
        plt.show()
    if (save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'Conductances', suffix))

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(var_dic['C_m'][:lim + 1])
    plt.ylabel('Capacitance')
    plt.title('Membrane')
    plt.subplot(4, 1, 2)
    plt.plot(var_dic['E_K'][:lim + 1], RATE_COLORS['n'])
    plt.ylabel('K E_rev')
    plt.subplot(4, 1, 3)
    plt.plot(var_dic['E_Ca'][:lim + 1], RATE_COLORS['e'])
    plt.ylabel('Ca E_rev')
    plt.subplot(4, 1, 4)
    plt.plot(var_dic['E_L'][:lim + 1], 'k')
    plt.ylabel('Leak E_rev')
    if (show):
        plt.show()
    if (save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'Membrane', suffix))

    plt.close('all')




def plot_vars_gate(name, mdp, scale, tau, suffix="", show=True, save=False):
    plt.figure()

    plt.subplot(3,1,1)
    plt.plot(mdp, 'r')
    plt.ylabel('Midpoint')
    plt.title(name)
    plt.subplot(3, 1, 2)
    plt.plot(scale, 'g')
    plt.ylabel('Scale')
    plt.subplot(3, 1, 3)
    plt.plot(tau)
    plt.ylabel('Time constant')

    if (show):
        plt.show()
    if(save):
        plt.savefig('%svar_%s_%s.png' % (DIR, name, suffix))


def plot_loss_rate(losses, rates, lim, suffix="", show=True, save=False):
    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(losses[:lim+1], 'r')
    plt.ylabel('Loss')

    plt.subplot(2,1,2)
    plt.plot(rates[:lim+1])
    plt.ylabel('Learning rate')

    if (show):
        plt.show()
    if(save):
        plt.savefig('%slosses_%s.png' % (DIR,suffix))
    plt.close()

def plots_output(ts, i_inj, cac_lum, y_cac_lum, suffix="", show=True, save=False):
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(ts, cac_lum, 'r')
    plt.ylabel('Ca2+ concentration predicted')

    plt.subplot(3, 1, 2)
    plt.plot(ts, y_cac_lum, 'r')
    plt.ylabel('Ca2+ concentration true')

    plt.subplot(3, 1, 3)
    plt.plot(ts, i_inj, 'k')
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
    plt.ylim(-1, 40)

    if (show):
        plt.show()

    if(save):
        plt.savefig('%soutput_%s.png' % (DIR,suffix))

def plots_output_double(ts, i_inj, v, y_v, cac, y_cac, suffix="", show=True, save=False):
    plt.figure()

    plt.subplot(3, 1, 2)
    plt.plot(ts, y_cac, 'g')
    plt.plot(ts, cac, 'r')
    plt.ylabel('Ca2+ concentration predicted')

    plt.subplot(3, 1, 1)
    plt.plot(ts, y_v, 'g')
    plt.plot(ts, v, 'r')
    plt.ylabel('Ca2+ concentration true')

    plt.subplot(3, 1, 3)
    plt.plot(ts, i_inj, 'k')
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

    if (show):
        plt.show()
    if(save):
        plt.savefig('%s%soutput_%s.png' % (DIR,IMG_DIR,suffix))
    plt.close()

"""plot i_ca and Ca conc depending on the voltage"""
def plots_ica_from_v(ts, V, results, suffix="", show=True, save=False):


    ica = results[:, 0]
    e = results[:, 1]
    f = results[:, 2]
    h = results[:, 3]
    cac = results[:, -1]

    plt.figure()

    plt.subplot(4, 1, 1)
    plt.title('Hodgkin-Huxley Neuron : I_ca from a fixed V')
    plt.plot(ts, ica, 'b')
    plt.ylabel('I_ca')

    plt.subplot(4, 1, 2)
    plt.plot(ts, cac, 'r')
    plt.ylabel('Ca2+ concentration')

    plt.subplot(4, 1, 3)
    plt.plot(ts, e, RATE_COLORS['e'], label='e')
    plt.plot(ts, f, RATE_COLORS['f'], label='f')
    plt.plot(ts, h, RATE_COLORS['h'], label='h')
    plt.ylabel('Gating Value')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(ts, V, 'k')
    plt.ylabel('V (input) (mV)')
    plt.xlabel('t (ms)')

    if (show):
        plt.show()
    if (save):
        plt.savefig('%sresults_%s.png' % (DIR, suffix))
    plt.close()


def plots_results_simp(ts, i_inj_values, results, suffix="", show=True, save=False):
    V = results[:, 0]
    cac = results[:, -1]

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.title('Hodgkin-Huxley Neuron')
    plt.plot(ts, V, 'k')
    plt.ylabel('V (mV)')

    plt.subplot(3, 1, 2)
    plt.plot(ts, cac, 'r')
    plt.ylabel('Ca2+ concentration')

    plt.subplot(3, 1, 3)
    plt.plot(ts, i_inj_values, 'k')
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
    plt.ylim(-1, 40)

    if(show):
        plt.show()

    if(save):
        plt.savefig('%sresults_%s.png'%(DIR,suffix))

"""plot all dynamics"""
def plots_results(model, ts, i_inj_values, results, suffix="", show=True, save=False):
    V = results[:, 0]
    p = results[:, 1]
    q = results[:, 2]
    n = results[:, 3]
    e = results[:, 4]
    f = results[:, 5]
    cac = results[:, 6]

    h = model.h(cac)
    ica = model.I_Ca(V, e, f, h)
    ik = model.I_Ks(V, n) + model.I_Kf(V, p, q)
    il = model.I_L(V)

    plt.figure()

    plt.subplot(5, 1, 1)
    plt.title('Hodgkin-Huxley Neuron')
    plt.plot(ts, V, 'k')
    plt.ylabel('V (mV)')

    plt.subplot(5, 1, 2)
    plt.plot(ts, cac, 'r')
    plt.ylabel('Ca2+ concentration')

    plt.subplot(5, 1, 3)
    plt.plot(ts, ica, 'c', label='$I_{Ca}$')
    plt.plot(ts, ik, 'y', label='$I_{K}$')
    plt.plot(ts, il, 'm', label='$I_{L}$')
    plt.ylabel('Current')
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(ts, p, RATE_COLORS['p'], label='p')
    plt.plot(ts, q, RATE_COLORS['q'], label='q')
    plt.plot(ts, n, RATE_COLORS['n'], label='n')
    plt.plot(ts, e, RATE_COLORS['e'], label='e')
    plt.plot(ts, f, RATE_COLORS['f'], label='f')
    plt.plot(ts, h, RATE_COLORS['h'], label='h')
    plt.ylabel('Gating Value')
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(ts, i_inj_values, 'k')
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
    # plt.ylim(-1, 40)

    if(show):
        plt.show()
    if(save):
        plt.savefig('%sresults_%s.png'%(DIR,suffix))
    plt.close()