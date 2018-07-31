"""
.. module:: 
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

import matplotlib as mpl
import socket
if (socket.gethostname()=='1080' or socket.gethostname()=='pixi'):
    mpl.use("Agg")
import numpy as np
import pickle
import pylab as plt
import pandas as pd
import seaborn as sns
import sys
import xml.etree.ElementTree as ET


from odin import utils
from odin import circuit as cr
from odin import coptim as co

dt = 0.1
n_parallel = 5

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
            if i % 20 == 0:
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

def get_curr():
    tree = ET.parse('c302_C2_FW.net.xml')
    root = tree.getroot()
    curs = np.zeros((50001, 39))
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


def show_res(dir):
    with open('forward_input', 'rb') as f:
        cur = pickle.load(f)
    cur = cur[:, np.newaxis, :]
    with open('forward_target', 'rb') as f:
        res = pickle.load(f)
    from odin import optim

    dir = utils.set_dir(dir)
    dic = optim.get_vars(dir, 620, loss=False)
    print(len(dic['G_gap']))
    print(np.unique(dic['G_gap'][:,0], return_counts=True))
    exit(0)
    # print(dic)
    dic = {v: np.array(val, dtype=np.float32) for v,val in dic.items()}
    ctf = cr.CircuitTf.create_random(n_neuron=39, syn_keys=syns_k, gap_keys=gaps_k,
                                     labels=labels, commands=commands, n_rand=dic['C_m'].shape[-1])
    # ctf = optim.get_model(dir)
    ctf.init_params = dic
    states = ctf.calculate(np.stack([cur for _ in range(dic['C_m'].shape[-1])], axis=-1))
    print(states.shape)
    for i in range(ctf.num):
        ctf.plots_output_mult(res[...,0], cur[:,0,i], states[...,i], suffix=i, show=True, save=True, trace=False)
    exit(0)

if __name__=='__main__':

    # get_conns()

    with open('forward_input', 'rb') as f:
        cur = pickle.load(f)

    df = pd.DataFrame(cur.transpose(), index=labels.values())
    sns.heatmap(df, cmap='jet')
    plt.title('Membrane potentials (mV)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    utils.save_show(False, False, 'Input Current', dpi=300)

    name = 'Forward_{}'.format(sys.argv[1])
    dir = utils.set_dir(name)

    with open('forward_target', 'rb') as f:
        res = pickle.load(f)

    print(cur.shape)
    print(res.shape)

    for i in range(4, len(labels)):
        print(i, labels[i], labels[i-1])
        res[:, i+1] = np.roll(res[:, i], 800, axis=0)
    for i in range(rev_labels['DD1'], rev_labels['VB11']+1):
        res[:7000, i+1] = -35.
    for i in range(rev_labels['VD1'], rev_labels['VD5']+1):
        res[:3000, i+1] = -35.
    for i in range(rev_labels['VD6'], rev_labels['VD13']+1):
        res[:2000, i+1] = -35.
    res = np.array([r + 1.5*np.random.randn(len(r)) for r in res])

    df = pd.DataFrame(res[:,1:].transpose(), index=labels.values(), columns=res[:,0])
    sns.heatmap(df, cmap='RdYlBu_r')
    plt.xticks([])
    plt.title('Membrane potentials (mV)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    utils.save_show(True, False, 'Target_Voltage', dpi=300)

    cur = cur[:, np.newaxis, :]
    res = res[:, np.newaxis, :]


    fixed = ()
    ctf = cr.CircuitTf.create_random(n_neuron=39, syn_keys=syns_k, gap_keys=gaps_k, groups=groups,
                                  labels=labels, commands=commands, n_rand=n_parallel, fixed=fixed)



    copt = co.CircuitOpt(circuit=ctf)
    print(res[...,1:].shape, cur.shape)
    copt.optimize(subdir=dir, train=[res[..., 0], cur, [res[..., 1:], None]], n_out=list(np.arange(39)), l_rate=(0.4, 9, 0.92))

