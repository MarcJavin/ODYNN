import pandas as pd

SYNS = 'Neuro279_Syn.csv'
GAPS = 'Neuro279_EJ.csv'
DATA = 'WBI.csv'
NAMES = ['AIBL', 'AIBR', 'ALA', 'ALNL', 'ALNR', 'AS10', 'ASKL', 'ASKR', 'AVAL',
       'AVAR', 'AVBL', 'AVBR', 'AVEL', 'AVER', 'AVFL', 'AVFR', 'BAGL', 'DA01',
       'DA07', 'DA09', 'DB01', 'DB02', 'DB07', 'DVA', 'DVC', 'LUAR', 'OLQDL',
       'OLQDR', 'OLQVL', 'OLQVR', 'PDA', 'PHAR', 'PVCL', 'PVCR', 'PVNL',
       'PVNR', 'RIBL', 'RIBR', 'RID', 'RIFR', 'RIMR', 'RIS', 'RIVR', 'RMED',
       'RMEL', 'RMEV', 'SABD', 'SABVL', 'SABVR', 'SIADL', 'SIADR', 'SIAVL',
       'SIAVR', 'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR', 'VA01', 'VA11', 'VA12',
       'VB02', 'VB11', 'VD11', 'VD13']

def get_synapses():
    return pd.read_csv(SYNS, index_col=0).loc[NAMES,NAMES]

def get_gaps():
    return pd.read_csv(GAPS, index_col=0).loc[NAMES,NAMES]

def get_data():
    return pd.read_csv(DATA, index_col=0)[NAMES]
