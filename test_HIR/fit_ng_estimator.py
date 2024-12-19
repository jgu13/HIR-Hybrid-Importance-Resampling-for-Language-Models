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
cache_dir = os.path.join(PROJ_HOME, 'cache_dir')

dsir = HashedNgramDSIR(raw_datasets, target_datasets, cache_dir=cache_dir, num_proc=1)
print("Fit NG importance estimator: ")     
start_time = time.time()
dsir.fit_ng_importance_estimator(num_tokens_to_fit='auto')
diff_time = time.time() - start_time
print("Time taken to fit ng importance estimator = {:.2f}".format(diff_time/60))
