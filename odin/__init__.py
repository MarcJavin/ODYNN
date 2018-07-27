import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Use on my server
import matplotlib as mpl
mpl.use("Agg")
from .utils import COLORS
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)
# Tune the plots appearance
import pylab as plt
SMALL_SIZE = 8
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
