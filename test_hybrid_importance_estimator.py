import time
print("------- Start here ------")
start_time = time.time()
from data_selection import HashedNgramDSIR
import os
import_time = time.time() - start_time
print("Import time taken = {:.2f} min".format(import_time / 60))

PROJ_HOME = os.path.expanduser("~/scratch/dsir")
data_path = os.path.join(PROJ_HOME, "tests")
raw_datasets = [os.path.join(os.path.join(data_path, "toy_data.jsonl"))]
target_datasets = [os.path.join(os.path.join(data_path, "toy_data_target.jsonl"))]

start_time = time.time()
dsir = HashedNgramDSIR(raw_datasets, target_datasets, cache_dir=os.path.join(PROJ_HOME, 'dsir_cache'), num_proc=1)
init_time = time.time() - start_time
print("Initialization DSIR takes {:.2f} min".format(init_time / 60))
# dsir.fit_ng_importance_estimator(num_tokens_to_fit='auto')
# start_time = time.time()
# dsir.fit_gmm_importance_estimator(num_components=1, 
#                                   raw_text_emb_path=dsir.cache_dir / "raw_text_emb.npy",
#                                   target_text_emb_path=dsir.cache_dir / "target_text_emb.npy")
# fit_gmm_time = time.time() - start_time
# print("Fit GMM takes {:.2f} min".format(fit_gmm_time / 60))
# dsir.compute_hybrid_importance_weight()