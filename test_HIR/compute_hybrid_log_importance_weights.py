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
cache_dir = '/home/mcb/users/jgu13/projects/HIR/cache_dir'
log_diff_path = '/home/mcb/users/jgu13/projects/HIR/cache_dir/ng_log_diff.npy'
ng_log_importance_weights_path = '/home/mcb/users/jgu13/projects/HIR/cache_dir/ng_log_importance_weights/ng_log_importance_weights_1420000.npy' # same for alpha = 0, 0.25, 0.75
gmm_log_importance_weights_path = '/home/mcb/users/jgu13/projects/HIR/cache_dir/output_alpha_0/gmm_log_importance_weights.npy' # same for alpha = 0, 0.25, 0.75

dsir = HashedNgramDSIR(raw_datasets, target_datasets, cache_dir=cache_dir, num_proc=1)
# alpha = 0
dsir.compute_hybrid_importance_weight(alpha = 0, 
                                      ng_log_importance_weights_path = ng_log_importance_weights_path,
                                      gmm_log_importance_weights_path = gmm_log_importance_weights_path,
                                      save_path = dsir.cache_dir / "hybrid_importance_weights_alpha_0.npy")
# alpha = 0.25
dsir.compute_hybrid_importance_weight(alpha = 0.25, 
                                      ng_log_importance_weights_path = ng_log_importance_weights_path,
                                      gmm_log_importance_weights_path = gmm_log_importance_weights_path,
                                      save_path = dsir.cache_dir / "hybrid_importance_weights_alpha_0.25.npy")
# alpha = 0.5
dsir.compute_hybrid_importance_weight(alpha = 0.5, 
                                      ng_log_importance_weights_path = ng_log_importance_weights_path,
                                      gmm_log_importance_weights_path = gmm_log_importance_weights_path,
                                      save_path = dsir.cache_dir / "hybrid_importance_weights_alpha_0.5.npy")
# # alpha = 0.75
dsir.compute_hybrid_importance_weight(alpha = 0.75, 
                                      ng_log_importance_weights_path = ng_log_importance_weights_path,
                                      gmm_log_importance_weights_path = gmm_log_importance_weights_path,
                                      save_path = dsir.cache_dir / "hybrid_importance_weights_alpha_0.75.npy")
# when alpha = 1, hybrid_importance_weights == ng_log_importance_weights