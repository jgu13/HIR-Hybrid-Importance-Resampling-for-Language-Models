#!/bin/bash

CACHE='/home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models/experimental/chache_dir'
ROOT_DIR='/home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models/experimental'
VIRTUAL_ENV='/path/to/.env'
PILE_PATH='/home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models/Pile'
DSIR_OUTPUT_DIR='/home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models/experimental/output'
PRETRAIN_OUTPUT_DIR='/home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models/experimental/model_outputdir'
WORD_VECTORS_PATH='/path/to/pretrained_fasttext_wordvecs.vec'
# Slurm
cluster_info='--partition <PARTITION_NAME>'

source $(conda info --base)/etc/profile.d/conda.sh
conda activate HIR
