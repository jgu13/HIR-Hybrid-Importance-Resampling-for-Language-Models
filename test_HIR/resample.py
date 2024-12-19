import sys
import numpy as np
sys.path.append('/home/mcb/users/jgu13/projects/HIR')

import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_selection.hashed_ngram_dsir import HashedNgramDSIR
import os
import time

PROJ_HOME = os.path.dirname(os.path.dirname(__file__))
raw_datasets = ['/home/mcb/users/jgu13/projects/HIR/Pile/Pile.jsonl']
target_datasets = ['/home/mcb/users/jgu13/projects/HIR/Pile/Target_set.jsonl']
cache_dir = os.path.join(PROJ_HOME, "cache_dir")
num_to_sample = 47563 # 1 / 30 of number of raw text samples, 1426891 

dsir = HashedNgramDSIR(raw_datasets, target_datasets, cache_dir=cache_dir, num_proc=1)
start_time = time.time()
# alpha = 0
log_importance_weights_path = dsir.cache_dir / "hybrid_log_importance_weights_alpha_0.npy"
dsir.resample(log_importance_weights_path=log_importance_weights_path,
              out_dir=os.path.join(PROJ_HOME, "outputdir/output_alpha_0"),
              num_to_sample=num_to_sample, # 1/30 of raw dataset
              top_k = True)
diff_time = time.time() - start_time
print(diff_time)
# # alpha = 0.25
# log_importance_weights_path = dsir.cache_dir / "hybrid_log_importance_weights_alpha_0.25.npy"
# dsir.resample(log_importance_weights_path=log_importance_weights_path,
#               out_dir=os.path.join(PROJ_HOME, "outputdir/output_alpha_0.25"),
#               num_to_sample=num_to_sample, # 1/30 of raw dataset
#               top_k = True)
# # alpha = 0.5
# log_importance_weights_path = dsir.cache_dir / "hybrid_log_importance_weights_alpha_0.5.npy"
# dsir.resample(log_importance_weights_path=log_importance_weights_path,
#               out_dir=os.path.join(PROJ_HOME, "outputdir/output_alpha_0.5"),
#               num_to_sample=num_to_sample, # 1/30 of raw dataset
#               top_k = True)
# # alpha = 0.75
# log_importance_weights_path = dsir.cache_dir / "hybrid_log_importance_weights_alpha_0.75.npy"
# dsir.resample(log_importance_weights_path=log_importance_weights_path,
#               out_dir=os.path.join(PROJ_HOME, "outputdir/output_alpha_0.75"),
#               num_to_sample=num_to_sample, # 1/30 of raw dataset
#               top_k = True)
