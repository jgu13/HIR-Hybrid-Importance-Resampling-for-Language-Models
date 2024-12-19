import sys
import numpy as np
sys.path.append('/home/mcb/users/jgu13/projects/HIR')

import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_selection.hashed_ngram_dsir import HashedNgramDSIR
import os
import time

PROJ_HOME = os.path.dirname(os.path.dirname(__file__))
raw_datasets = [os.path.join(PROJ_HOME,'Pile/Pile.jsonl')]
target_datasets = [os.path.join(PROJ_HOME, 'Pile/Target_set.jsonl')]
cache_dir = os.path.join(PROJ_HOME, 'cache_dir/output_alpha_0')
checkpoint_dir = os.path.join(PROJ_HOME, 'checkpoints/alpha_0')
gmm_raw_checkpoint_path = os.path.join(PROJ_HOME, 'checkpoints/alpha_0/gmm_raw_final.pkl')

dsir = HashedNgramDSIR(raw_datasets, target_datasets, cache_dir=cache_dir, num_proc=1)
print("Fit GMM importance estimator: ")     
start_time = time.time()
dsir.fit_gmm_importance_estimator(raw_max_samples=1426891, 
                                  target_max_samples=536997,
                                  chunk_size=10000,
                                raw_text_emb_path = '/home/mcb/users/jgu13/projects/HIR/dsir_cache/raw_text_embedding',
                                target_text_emb_path = '/home/mcb/users/jgu13/projects/HIR/dsir_cache/target_set_embedding', 
                                checkpoint_dir=checkpoint_dir,
                                gmm_raw_checkpoint_path=gmm_raw_checkpoint_path)
diff = time.time() - start_time
print("Time fit gmm importance estimator = {:.2f} min".format(diff/60))