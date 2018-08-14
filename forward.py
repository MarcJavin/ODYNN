"""
.. module:: 
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""




from odynn import utils, neuron
from odynn import circuit as cr
from odynn import coptim as co

from odynn.models import cfg_model
import numpy as np
import pickle
import pylab as plt
import pandas as pd
import seaborn as sns
import sys
import xml.etree.ElementTree as ET

dt = 0.1
n_parallel = 5
fake = True
eq_cost = True

labels = {0: 'AVBL',
              1: 'AVBR',
              2: 'DB1',
              3: 'DB2',
              4: 'DB3',
              5: 'DB4',
              6: 'DB5',
              7: 'DB6',
              8: 'DB7',
              9: 'DD1',
              10: 'DD2',
              11: 'DD3',
              12: 'DD4',
              13: 'DD5',
              14: 'DD6',
              15: 'VB1',
              16: 'VB2',
              17: 'VB3',
              18: 'VB4',
              19: 'VB5',
              20: 'VB6',
              21: 'VB7',
              22: 'VB8',
              23: 'VB9',
              24: 'VB10',
              25: 'VB11',
              26: 'VD1',
              27: 'VD2',
              28: 'VD3',
              29: 'VD4',
              30: 'VD5',
              31: 'VD6',
              32: 'VD7',
              33: 'VD8',
              34: 'VD9',
              35: 'VD10',
              36: 'VD11',
              37: 'VD12',
              38: 'VD13'
              }
rev_labels = {v: k for k,v in labels.items()}
groups = [0,0] + [1 for _ in range(7)] + [2 for _ in range(6)] + [3 for _ in range(11)] + [4 for _ in range(13)]

commands = [labels.values()]
commands = set(commands[2:])

DEFAULT_F = {
    'decay_ca': 13.8,
    'rho_ca': 0.23,
    'p__tau': 2.25,  # ms
    'p__scale': 7.43,  # mV
    'p__mdp': -8.05,  # mV
    'q__tau': 150.,
    'q__scale': -9.97,
    'q__mdp': -15.6,
    'n__tau': 25.,
    'n__scale': 15.85,
    'n__mdp': 19.9,
    'f__tau': 151.,
    'f__scale': -5.,
    'f__mdp': 25.2,
    'e__tau': 0.1,
    'e__scale': 6.75,
    'e__mdp': -3.36,
    'h__alpha': 0.282,  # None
    'h__scale': -1.,  # mol per m3
    'h__mdp': 6.4,
    'C_m': 5.0,
    'g_Ca': 1.81,
    'g_Ks': 0.46,
    'g_Kf': 0.042,
    'g_L': 0.002,
    'E_Ca': 10.0,
    'E_K': -70.0,
    'E_L': -60.0
}

gaps_k = [
    (rev_labels['AVBL'], rev_labels['AVBR']),
    (rev_labels['AVBL'], rev_labels['DB2']),
    (rev_labels['AVBL'], rev_labels['DB3']),
    (rev_labels['AVBL'], rev_labels['DB4']),
    (rev_labels['AVBL'], rev_labels['DB5']),
    (rev_labels['AVBL'], rev_labels['DB6']),
    (rev_labels['AVBL'], rev_labels['DB7']),
    (rev_labels['AVBL'], rev_labels['VB1']),
    (rev_labels['AVBL'], rev_labels['VB2']),
    (rev_labels['AVBL'], rev_labels['VB4']),
    (rev_labels['AVBL'], rev_labels['VB5']),
    (rev_labels['AVBL'], rev_labels['VB6']),
    (rev_labels['AVBL'], rev_labels['VB7']),
    (rev_labels['AVBL'], rev_labels['VB8']),
    (rev_labels['AVBL'], rev_labels['VB9']),
    (rev_labels['AVBL'], rev_labels['VB10']),
    (rev_labels['AVBL'], rev_labels['VB11']),
    (rev_labels['AVBR'], rev_labels['DB1']),
    (rev_labels['AVBR'], rev_labels['DB2']),
    (rev_labels['AVBR'], rev_labels['DB3']),
    (rev_labels['AVBR'], rev_labels['DB4']),
    (rev_labels['AVBR'], rev_labels['DB5']),
    (rev_labels['AVBR'], rev_labels['DB6']),
    (rev_labels['AVBR'], rev_labels['DB7']),
    (rev_labels['AVBR'], rev_labels['VB2']),
    (rev_labels['AVBR'], rev_labels['VB3']),
    (rev_labels['AVBR'], rev_labels['VB4']),
    (rev_labels['AVBR'], rev_labels['VB5']),
    (rev_labels['AVBR'], rev_labels['VB6']),
    (rev_labels['AVBR'], rev_labels['VB7']),
    (rev_labels['AVBR'], rev_labels['VB8']),
    (rev_labels['AVBR'], rev_labels['VB9']),
    (rev_labels['AVBR'], rev_labels['VB10']),
    (rev_labels['AVBR'], rev_labels['VB11']),
    (rev_labels['DB1'], rev_labels['DB2']),
    (rev_labels['DB1'], rev_labels['VB3']),
    (rev_labels['DB2'], rev_labels['DB3']),
    (rev_labels['DB2'], rev_labels['VB4']),
    (rev_labels['DB3'], rev_labels['DB4']),
    (rev_labels['DB3'], rev_labels['DD3']),
    (rev_labels['DB3'], rev_labels['VB6']),
    (rev_labels['DB4'], rev_labels['DB5']),
    (rev_labels['DB4'], rev_labels['DD3']),
    (rev_labels['DB5'], rev_labels['DB6']),
    (rev_labels['DB6'], rev_labels['DB7']),
    (rev_labels['VB1'], rev_labels['VB2']),
    (rev_labels['VB2'], rev_labels['VB3']),
    (rev_labels['VB3'], rev_labels['VB4']),
    (rev_labels['VB4'], rev_labels['VB5']),
    (rev_labels['VB5'], rev_labels['VB6']),
    (rev_labels['VB6'], rev_labels['VB7']),
    (rev_labels['VB7'], rev_labels['VB8']),
    (rev_labels['VB8'], rev_labels['VB9']),
    (rev_labels['VB9'], rev_labels['VB10']),
    (rev_labels['VB10'], rev_labels['VB11']),
    (rev_labels['VB10'], rev_labels['VD11']),
    (rev_labels['VB10'], rev_labels['VD12']),
    (rev_labels['DD1'], rev_labels['DD2']),
    (rev_labels['DD1'], rev_labels['VD2']),
    (rev_labels['DD1'], rev_labels['VD3']),
    (rev_labels['DD2'], rev_labels['DD3']),
    (rev_labels['DD3'], rev_labels['DD4']),
    (rev_labels['DD4'], rev_labels['DD5']),
    (rev_labels['DD2'], rev_labels['VD3']),
    (rev_labels['DD2'], rev_labels['VD4']),
    (rev_labels['DD2'], rev_labels['VD5']),
    (rev_labels['DD5'], rev_labels['DD6']),
    (rev_labels['DD6'], rev_labels['VD11']),
    (rev_labels['DD6'], rev_labels['VD12']),
    (rev_labels['DD6'], rev_labels['VD13']),
    (rev_labels['VD1'], rev_labels['VD2']),
    (rev_labels['VD2'], rev_labels['VD3']),
    (rev_labels['VD3'], rev_labels['VD4']),
    (rev_labels['VD4'], rev_labels['VD5']),
    (rev_labels['VD5'], rev_labels['VD6']),
    (rev_labels['VD6'], rev_labels['VD7']),
    (rev_labels['VD7'], rev_labels['VD8']),
    (rev_labels['VD8'], rev_labels['VD9']),
    (rev_labels['VD9'], rev_labels['VD10']),
    (rev_labels['VD10'], rev_labels['VD11']),
    (rev_labels['VD11'], rev_labels['VD12']),
    (rev_labels['VD12'], rev_labels['VD13'])
]

syns_k = {(rev_labels['AVBL'], rev_labels['AVBR']): True,
(rev_labels['AVBL'], rev_labels['VB2']): True,
(rev_labels['AVBR'], rev_labels['AVBL']): True,
(rev_labels['AVBR'], rev_labels['DB4']): True,
(rev_labels['AVBR'], rev_labels['VD3']): True,
(rev_labels['DB1'], rev_labels['DD1']): False,
(rev_labels['DB1'], rev_labels['VD2']): True,
(rev_labels['DB1'], rev_labels['VD3']): True,
(rev_labels['DB2'], rev_labels['DD1']): False,
(rev_labels['DB2'], rev_labels['DD2']): False,
(rev_labels['DB2'], rev_labels['VD3']): True,
(rev_labels['DB2'], rev_labels['VD4']): True,
(rev_labels['DB3'], rev_labels['DD2']): False,
(rev_labels['DB3'], rev_labels['DD3']): False,
(rev_labels['DB3'], rev_labels['DD5']): False,
(rev_labels['DB3'], rev_labels['VD4']): True,
(rev_labels['DB3'], rev_labels['VD5']): True,
(rev_labels['DB3'], rev_labels['VD6']): True,
(rev_labels['DB4'], rev_labels['DD3']): False,
(rev_labels['DB4'], rev_labels['DD4']): False,
(rev_labels['DB4'], rev_labels['DD5']): False,
(rev_labels['DB4'], rev_labels['DD6']): False,
(rev_labels['DB4'], rev_labels['VD6']): True,
(rev_labels['DB5'], rev_labels['VD4']): True,
(rev_labels['DB5'], rev_labels['VD5']): True,
(rev_labels['DB5'], rev_labels['VD6']): True,
(rev_labels['DB5'], rev_labels['VD7']): True,
(rev_labels['DB6'], rev_labels['VD7']): True,
(rev_labels['DB6'], rev_labels['VD8']): True,
(rev_labels['DB6'], rev_labels['VD9']): True,
(rev_labels['DB7'], rev_labels['VD8']): True,
(rev_labels['DB7'], rev_labels['VD9']): True,
(rev_labels['DB7'], rev_labels['VD10']): True,
(rev_labels['DB7'], rev_labels['VD11']): True,
(rev_labels['DB7'], rev_labels['VD12']): True,
(rev_labels['DB7'], rev_labels['VD13']): True,
(rev_labels['DD1'], rev_labels['VB2']): False,
(rev_labels['DD1'], rev_labels['VD2']): False,
(rev_labels['DD2'], rev_labels['VD3']): False,
(rev_labels['DD2'], rev_labels['VD4']): False,
(rev_labels['VB1'], rev_labels['DD1']): True,
(rev_labels['VB1'], rev_labels['VD1']): False,
(rev_labels['VB1'], rev_labels['VD2']): False,
(rev_labels['VB2'], rev_labels['DD1']): True,
(rev_labels['VB2'], rev_labels['DD2']): True,
(rev_labels['VB2'], rev_labels['VD2']): False,
(rev_labels['VB2'], rev_labels['VD3']): False,
(rev_labels['VB3'], rev_labels['DD2']): True,
(rev_labels['VB3'], rev_labels['VD3']): False,
(rev_labels['VB3'], rev_labels['VD4']): False,
(rev_labels['VB4'], rev_labels['DD2']): True,
(rev_labels['VB4'], rev_labels['DD3']): True,
(rev_labels['VB4'], rev_labels['VD4']): False,
(rev_labels['VB4'], rev_labels['VD5']): False,
(rev_labels['VB5'], rev_labels['DD3']): True,
(rev_labels['VB5'], rev_labels['VD6']): False,
(rev_labels['VB6'], rev_labels['DD4']): True,
(rev_labels['VB6'], rev_labels['VD6']): False,
(rev_labels['VB6'], rev_labels['VD7']): False,
(rev_labels['VB7'], rev_labels['DD5']): True,
(rev_labels['VB7'], rev_labels['VD8']): False,
(rev_labels['VB7'], rev_labels['VD9']): False,
(rev_labels['VB8'], rev_labels['DD5']): True,
(rev_labels['VB8'], rev_labels['DD6']): True,
(rev_labels['VB8'], rev_labels['VD9']): False,
(rev_labels['VB8'], rev_labels['VD10']): False,
(rev_labels['VB9'], rev_labels['DD5']): True,
(rev_labels['VB9'], rev_labels['DD6']): True,
(rev_labels['VB9'], rev_labels['VD10']): False,
(rev_labels['VB9'], rev_labels['VD11']): False,
(rev_labels['VB10'], rev_labels['DD6']): True,
(rev_labels['VB10'], rev_labels['VD11']): False,
(rev_labels['VB10'], rev_labels['VD12']): False,
(rev_labels['VB11'], rev_labels['DD6']): True,
(rev_labels['VB11'], rev_labels['VD12']): False,
(rev_labels['VB11'], rev_labels['VD13']): False,
(rev_labels['VD1'], rev_labels['DD1']): False,
(rev_labels['VD1'], rev_labels['VB1']): False,
(rev_labels['VD2'], rev_labels['DD1']): False,
(rev_labels['VD2'], rev_labels['VB2']): False,
(rev_labels['VD3'], rev_labels['DD1']): False,
(rev_labels['VD3'], rev_labels['VB2']): False,
(rev_labels['VD3'], rev_labels['VB3']): False,
(rev_labels['VD4'], rev_labels['VB3']): False,
(rev_labels['VD5'], rev_labels['VB1']): False,
(rev_labels['VD5'], rev_labels['VB4']): False,
(rev_labels['VD6'], rev_labels['VB5']): False,
(rev_labels['VD7'], rev_labels['VB6']): False,
(rev_labels['VD7'], rev_labels['VB7']): False,
(rev_labels['VD8'], rev_labels['VB7']): False,
(rev_labels['VD9'], rev_labels['VB8']): False,
(rev_labels['VD10'], rev_labels['VB9']): False,
(rev_labels['VD11'], rev_labels['DD6']): False,
(rev_labels['VD11'], rev_labels['VB10']): False,
(rev_labels['VD12'], rev_labels['DD6']): False,
(rev_labels['VD12'], rev_labels['VB11']): False,
(rev_labels['VD13'], rev_labels['DD6']): False,
(rev_labels['VD13'], rev_labels['VD12']): False,
(rev_labels['DB1'], rev_labels['DB2']): True,
(rev_labels['DB2'], rev_labels['DB3']): True,
(rev_labels['DB3'], rev_labels['DB4']): True,
(rev_labels['DB4'], rev_labels['DB5']): True,
(rev_labels['DB5'], rev_labels['DB6']): True,
(rev_labels['DB6'], rev_labels['DB7']): True,
(rev_labels['VB1'], rev_labels['VB2']): True,
(rev_labels['VB2'], rev_labels['VB3']): True,
(rev_labels['VB3'], rev_labels['VB4']): True,
(rev_labels['VB4'], rev_labels['VB5']): True,
(rev_labels['VB5'], rev_labels['VB6']): True,
(rev_labels['VB6'], rev_labels['VB7']): True,
(rev_labels['VB7'], rev_labels['VB8']): True,
(rev_labels['VB8'], rev_labels['VB9']): True,
(rev_labels['VB9'], rev_labels['VB10']): True,
(rev_labels['VB10'], rev_labels['VB11']): True}

def get_data():
    with open('data/c302_C2_FW.dat', 'r') as f:
        res = []
        for i, line in enumerate(f):
            if i % (dt / 0.005) == 0:
                state = line.split()
                state = np.array([float(st) for st in state])
                res.append(state)
    res = np.array(res)

    res *= 1000
    temp = res[:, 16:17]
    res[:, 16:17] = res[:, 24:25]
    res[:, 24:25] = temp
    temp = res[:, 27:30]
    res[:, 27:30] = res[:, 35:38]
    res[:, 35:38] = temp

    with open('forward_target', 'wb') as f:
        pickle.dump(res, f)
    print(res.shape)
    return res

def get_curr():
    tree = ET.parse('c302_C2_FW.net.xml')
    root = tree.getroot()
    curs = np.zeros((int(5000/dt) +1, 39))
    for el in root:
        if el.tag == '{http://www.neuroml.org/schema/neuroml2}pulseGenerator':
            name = el.attrib['id'].split('_')[1]
            try:
                start = int(el.attrib['delay'][:-2])
                try:
                    dur = int(el.attrib['duration'][:-2])
                except:
                    dur = int(el.attrib['duration'][:-4])
                val = int(el.attrib['amplitude'][:-2])
                num = rev_labels[name]
                st = int(start / dt)
                end = int((start + dur) / dt)
                curs[st:end, num] = val
            except:
                pass
    with open('forward_input', 'wb') as f:
        pickle.dump(curs, f)
    print(curs.shape)
    return curs


def get_conns():
    tree = ET.parse('c302_C2_FW.net.xml')
    root = tree.getroot()
    for el in root:
        if el.tag == '{http://www.neuroml.org/schema/neuroml2}network':
            for c in el:
                if c.tag == '{http://www.neuroml.org/schema/neuroml2}electricalProjection':
                    for cc in c:
                        if 'Muscle' not in cc.attrib['postCell']:
                            print('(rev_labels[\'%s\'], rev_labels[\'%s\']),' % (c.attrib['presynapticPopulation'], c.attrib['postsynapticPopulation']))
                # if c.tag == '{http://www.neuroml.org/schema/neuroml2}continuousProjection':
                #     for cc in c:
                        # if 'Muscle' not in cc.attrib['postCell']:
                        #     print('(rev_labels[\'%s\'], rev_labels[\'%s\']): %s,' % (
                        #     c.attrib['presynapticPopulation'], c.attrib['postsynapticPopulation'], 'exc' in cc.attrib['postComponent']))


def show_res(dir, j=-1):
    # with open('forward_input', 'rb') as f:
    #     cur = pickle.load(f)
    # cur = cur[:, np.newaxis, :]
    from odynn import optim

    dir = utils.set_dir(dir)
    train, __ = optim.get_data(dir)
    cur = train[1]
    dic = optim.get_vars(dir, j, loss=False)
    [print(k) for k in dic.keys()]
    # print(dic)
    dic = {v: np.array(val, dtype=np.float32) for v,val in dic.items()}
    # ctf = cr.CircuitTf.create_random(n_neuron=39, syn_keys=syns_k, gap_keys=gaps_k,
    #                                  labels=labels, commands=commands, n_rand=5, fixed='all')
    ctf = optim.get_model(dir)
    # print(ctf._neurons.parameter_names)
    dic.update(ctf._neurons.init_params)
    dic['tau'] = dic['tau'] * 100
    # ctf._neurons.init_names()
    ctf.init_params = dic
    states = ctf.calculate(np.stack([cur for _ in range(ctf.num)], axis=-1))
    print(states.shape)
    for i in range(ctf.num):
        try:
            ctf.plots_output_mult(train[0], cur[:,0,i], states[...,i], suffix='%s_epoch%s'%(i,j), show=True, save=True, trace=False)
        except:
            print('Fail')
    exit(0)

def count_in_out():
    count = np.zeros((39,3))
    for s in syns_k.keys():
        count[s[0], 0] += 1
        count[s[1], 1] += 1
    for g in gaps_k:
        count[g[0], 2] += 1
        count[g[1], 2] += 1
    w = [1+(count[i,0] + count[i,2]/2)/(count[i,1] + 1) for i in range(39)]
    for i in range(39):
        print(labels[i], count[i], w[i])
    return w

if __name__=='__main__':
    show_res('Forward_celegtestfakeeqcost0.5', 300)

    get_data()
    get_curr()

    with open('forward_input', 'rb') as f:
        cur = pickle.load(f)

    df = pd.DataFrame(cur.transpose(), index=labels.values())
    sns.heatmap(df, cmap='jet')
    plt.title('Membrane potentials (mV)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    utils.save_show(False, False, 'Input Current', dpi=300)

    suffix = sys.argv[1]
    for i,m in enumerate(cfg_model.models):
        if cfg_model.NEURON_MODEL == m:
            suffix = cfg_model.models_name[i] + suffix
            break
    if fake:
        suffix = suffix + 'fake'
    else:
        suffix = suffix + 'real'
    if eq_cost:
        suffix = suffix + 'eqcost'
    else:
        suffix = suffix + 'difcost'
    suffix = suffix + str(dt)
    name = 'Forward_{}'.format(suffix)

    dir = utils.set_dir(name)

    with open(dir+'settings_fw', 'w') as f:
        f.write('eq_cost : %s'%eq_cost + '\n' +
                'fake : %s'%fake + '\n' +
                'model %s'%cfg_model.NEURON_MODEL)

    with open('forward_target', 'rb') as f:
        res = pickle.load(f)

    print(cur.shape)
    print(res.shape)

    if(fake):
        for i in range(4, len(labels)):
            res[:, i+1] = np.roll(res[:, i], int(80/dt), axis=0)
        # for i in range(rev_labels['DD1'], rev_labels['VB11']+1):
        #     res[:7000, i+1] = -35.
        # for i in range(rev_labels['VD1'], rev_labels['VD5']+1):
        #     res[:3000, i+1] = -35.
        # for i in range(rev_labels['VD6'], rev_labels['VD13']+1):
        #     res[:2000, i+1] = -35.
        res[:,1:] = np.array([r + 1.*np.random.randn(len(r)) for r in res[:,1:]])

    df = pd.DataFrame(res[:,1:].transpose(), index=labels.values(), columns=res[:,0])
    sns.heatmap(df, cmap='RdYlBu_r')
    plt.xticks([])
    plt.title('Membrane potentials (mV)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    utils.save_show(False, True, 'Target_Voltage', dpi=300)

    cur = cur[:, np.newaxis, :]
    res = res[:, np.newaxis, :]


    fixed = ()
    neurons = neuron.BioNeuronTf([DEFAULT_F for _ in range(39)], fixed='all', dt=dt)
    ctf = cr.CircuitTf.create_random(n_neuron=39, neurons=neurons, syn_keys=syns_k, dt=dt, gap_keys=gaps_k, groups=groups,
                                  labels=labels, commands=commands, n_rand=n_parallel, fixed=fixed)



    copt = co.CircuitOpt(circuit=ctf)
    print(res[...,1:].shape, cur.shape)
    if eq_cost:
        w_n = None
    else:
        w_n = count_in_out()
    copt.optimize(subdir=dir, train=[res[..., 0], cur, [res[..., 1:], None]], w_n=count_in_out(), n_out=list(np.arange(39)), l_rate=(0.4, 9, 0.92))

