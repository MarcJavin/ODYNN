import pandas as pd
from os.path import dirname
HERE = dirname(__file__)

SYNS = '/Neuro279_Syn.csv'
GAPS = '/Neuro279_EJ.csv'
def dataname(i):
    return '/WBInorm%s.csv'%i
DERIV = '/dWBIdt.csv'
N_REC = 6
NAMES = ['AFDL', 'AIBL', 'AIBR', 'ALA', 'ALNL', 'ALNR', 'AS10', 'ASKL', 'ASKR', 'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVEL', 'AVER', 'AVFL', 'AVFR', 'AWAR', 'AWCL', 'AWCR', 'BAGL', 'BAGR', 'DA01', 'DA07', 'DA09', 'DB01', 'DB02', 'DB07', 'DVA', 'DVB', 'DVC', 'LUAL', 'LUAR', 'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR', 'PDA', 'PHAL', 'PHAR', 'PLML', 'PVCL', 'PVCR', 'PVNL', 'PVNR', 'RIBL', 'RIBR', 'RID', 'RIFR', 'RIML', 'RIMR', 'RIS', 'RIVL', 'RIVR', 'RMED', 'RMEL', 'RMEV', 'SABD', 'SABVL', 'SABVR', 'SIADL', 'SIADR', 'SIAVL', 'SIAVR', 'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR', 'URADR', 'URAVL', 'URAVR', 'URYDR', 'URYVL', 'URYVR', 'VA01', 'VA11', 'VA12', 'VB01', 'VB02', 'VB11', 'VD11', 'VD13']

def get_synapses():
    return pd.read_csv(HERE+SYNS, index_col=0).loc[NAMES,NAMES]

def get_gaps():
    return pd.read_csv(HERE+GAPS, index_col=0).loc[NAMES,NAMES]

def get_data(i=0):
    return pd.read_csv(HERE+dataname(i), index_col=0)#[NAMES]

def get_all_data():
    datasets = []
    for i in range(N_REC):
        datasets.append(get_data(i))
    return datasets

def get_all_names():
    all_id = set()
    for i in range(N_REC):
        all_id = all_id.union(set(get_data(i).columns))
    all_names = [n  for n in all_id if any(c.isalpha() for c in n)]
    all_names = sorted(all_names)[:-3]
    all_names.remove('AS10.1')
    return all_names

def get_deriv():
    return pd.read_csv(HERE+DERIV, index_col=0)[NAMES]
