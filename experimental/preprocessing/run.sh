#!/bin/bash
source config.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda activate HIR

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE

ARGS=$1

python reformat_and_chunk_data.py ${ARGS}


