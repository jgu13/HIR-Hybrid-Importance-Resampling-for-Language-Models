#!/bin/bash

source config.sh

LOGDIR=logs/preprocessing/qualitystats
mkdir -p ${LOGDIR}

# Process subsets sequentially
for SUBSET in 00; do
    echo "Processing quality stats for subset ${SUBSET}..."
    bash run_quality_stats.sh "${PILE_PATH}/chunked/${SUBSET}_128/${SUBSET}_128.json" \
        > ${LOGDIR}/chunk_${SUBSET}.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Error processing subset ${SUBSET}. Check log: ${LOGDIR}/chunk_${SUBSET}.log"
        exit 1
    fi
done
echo "All subsets processed successfully."

# Process validation data
echo "Processing quality stats for validation data..."
bash run_quality_stats.sh "${PILE_PATH}/chunked/VAL_128/val_128.json" \
    > ${LOGDIR}/chunk_val.log 2>&1
if [ $? -ne 0 ]; then
    echo "Error processing validation data. Check log: ${LOGDIR}/chunk_val.log"
    exit 1
fi
echo "Validation data processed successfully."

echo "All quality stats tasks completed."
