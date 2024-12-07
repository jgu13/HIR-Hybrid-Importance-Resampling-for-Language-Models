import time
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_selection.hashed_ngram_dsir import HashedNgramDSIR
import os

PROJ_HOME = os.path.dirname(__file__)
data_path = os.path.join(PROJ_HOME, "tests")
raw_datasets = [os.path.join(os.path.join(data_path, "toy_data.jsonl"))]
target_datasets = [os.path.join(os.path.join(data_path, "toy_data_target.jsonl"))]

dsir = HashedNgramDSIR(raw_datasets, target_datasets, cache_dir=os.path.join(PROJ_HOME, 'dsir_cache'), num_proc=1)
dsir.fit_ng_importance_estimator(num_tokens_to_fit='auto')
dsir.fit_gmm_importance_estimator()    
# uncomment to run fit_gmm_importance_estimator with pre-computed raw and text embeddings      
# dsir.fit_gmm_importance_estimator(raw_text_emb_path = dsir.cache_dir / "raw text embeddings",
#                                 target_text_emb_path = dsir.cache_dir / "target text embeddings")
dsir.compute_hybrid_importance_weight(alpha = 0.5, save_path = dsir.cache_dir / "hybrid_importance_weights_alpha_0.5.npy") # TODO: try alpha = [0.0, 0.25, 0.5, 0.75, 1.0]