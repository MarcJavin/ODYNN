import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Use on my server
import matplotlib as mpl
import socket
if (socket.gethostname()=='1080'):
    mpl.use("Agg")
