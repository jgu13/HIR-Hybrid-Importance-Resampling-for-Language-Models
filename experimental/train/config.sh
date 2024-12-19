#!/bin/bash
ln -s /home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models HIR


CACHE='HIR/experimental/chache_dir'
ROOT_DIR='HIR/experimental'
VIRTUAL_ENV='/path/to/.env'
PILE_PATH='HIR/Pile'
DSIR_OUTPUT_DIR='HIR/experimental/output'
PRETRAIN_OUTPUT_DIR='HIR/experimental/model_outputdir'
WORD_VECTORS_PATH='/path/to/pretrained_fasttext_wordvecs.vec'
# Slurm
cluster_info='--partition <PARTITION_NAME>'

source $(conda info --base)/etc/profile.d/conda.sh
conda activate HIR
