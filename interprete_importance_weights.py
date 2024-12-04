import numpy as np
import os.path as osp

PROJ_HOME = osp.expanduser("~/scratch/dsir")
cache_dir = osp.join(PROJ_HOME, "dsir_cache", "log_importance_weights")

res = np.load(osp.join(cache_dir, "0.npy"))

print(res)