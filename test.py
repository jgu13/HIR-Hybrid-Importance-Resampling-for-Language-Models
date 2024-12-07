from data_selection import HashedNgramDSIR
import os

PROJ_HOME = os.path.expanduser("~/scratch/dsir")
data_path = os.path.join(PROJ_HOME, "tests")
raw_datasets = [os.path.join(os.path.join(data_path, "toy_data.jsonl"))]
target_datasets = [os.path.join(os.path.join(data_path, "toy_data_target.jsonl"))]

dsir = HashedNgramDSIR(raw_datasets, target_datasets, cache_dir=os.path.join(PROJ_HOME, 'dsir_cache'), num_proc=1)
dsir.fit_ng_importance_estimator(num_tokens_to_fit='auto')
dsir.compute_importance_weights()

# dsir.resample(out_dir='resampled', num_to_sample=10000000, cache_dir='/path/to/resampled_cache')