# wd = "/wholebrain/scratch/areaxfs/"
wd = "/u/pschuber/areaxfs/"
import os
import socket
#
# # define global working directory
# # wd = "/u/pschuber/areaxfs/"
sn = socket.gethostname()
if sn == "L-N1-050" or sn == "fordprefect":
    # wd = "/home/pschuber/mnt/external2/j0126_dense_syconn_v1/"
    wd = "/home/pschuber/mnt/wb/wholebrain/u/pschuber/areaxfs/"
wd = os.path.expanduser(wd)
