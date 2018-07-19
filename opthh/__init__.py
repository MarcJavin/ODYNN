import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Use on my server
import matplotlib as mpl
import socket
if (socket.gethostname()=='1080' or socket.gethostname()=='pixi'):
    mpl.use("Agg")
from .utils import COLORS
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)
# Tune the plots appearance
import pylab as plt
SMALL_SIZE = 8
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
