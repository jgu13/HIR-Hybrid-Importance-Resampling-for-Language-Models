#!/bin/bash

source config.sh

LOGDIR=logs/preprocess
mkdir -p ${LOGDIR}

# Process subsets sequentially
for SUBSET in 00; do
    echo "Processing subset ${SUBSET}..."
    bash run.sh "--input_filename ${SUBSET}.jsonl.zst --chunk_length 128 --input_dir ${PILE_PATH} --output_dir ${PILE_PATH}/chunked/${SUBSET}_128 --cache_dir ${CACHE}" \
        > ${LOGDIR}/chunk_${SUBSET}.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Error processing subset ${SUBSET}. Check log: ${LOGDIR}/chunk_${SUBSET}.log"
        exit 1
    fi
done
echo "All subsets processed successfully."

# Process validation data
echo "Processing validation data..."
bash run.sh "--input_filename val.jsonl.zst --chunk_length 128 --input_dir ${PILE_PATH} --output_dir ${PILE_PATH}/chunked/VAL_128 --cache_dir ${CACHE}" \
    > ${LOGDIR}/chunk_val.log 2>&1
if [ $? -ne 0 ]; then
    echo "Error processing validation data. Check log: ${LOGDIR}/chunk_val.log"
    exit 1
fi
echo "Validation data processed successfully."

echo "All preprocessing tasks completed."
