#!/bin/bash
source config.sh

# Input parameters
task=$1
run_name=$2
other_args=$3
num_to_retrieve=${4:-25000000}

# Log directory setup
LOGDIR=logs/data_selection/dsir/${run_name}
mkdir -p ${LOGDIR}

# Number of chunks to process
NUM_CHUNKS=1

# # Step 1: Prepare chunks sequentially
# echo "Step 1: Preparing chunks..."
# for CHUNK_IDX in 00; do
#     echo "Preparing chunk ${CHUNK_IDX}/${NUM_CHUNKS}..."
#     bash run_dsir_helper.sh ${task} "--pipeline_step prepare --chunk_idx ${CHUNK_IDX} --num_chunks ${NUM_CHUNKS} ${other_args} --num_proc 16" \
#         > ${LOGDIR}/prepare_${CHUNK_IDX}.log 2>&1
# done
# echo "Chunk preparation complete."

# # Step 2: Calculate importance weights sequentially
# echo "Step 2: Calculating importance weights..."
# for CHUNK_IDX in 00; do
#     echo "Processing chunk ${CHUNK_IDX}/${NUM_CHUNKS} for importance weights..."
#     bash run_dsir_helper.sh ${task} "--pipeline_step importance_weights --chunk_idx ${CHUNK_IDX} --num_chunks ${NUM_CHUNKS} ${other_args} --num_proc 16" \
#         > ${LOGDIR}/predict_${CHUNK_IDX}.log 2>&1
# done
# echo "Importance weight calculation complete."

# Step 3: Resampling
echo "Step 3: Resampling..."
bash run_dsir_helper.sh ${task} "--pipeline_step resample ${other_args} --num_proc 8 --num_to_retrieve ${num_to_retrieve}" \
    > ${LOGDIR}/retrieve_${num_to_retrieve}.log 2>&1
echo "Resampling complete."
