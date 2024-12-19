#!/bin/bash
set -x

source config.sh

# Input parameters
TASK=$1
SUFFIX=$2
PORT=$3
PRETRAIN_DATA_PATH=$4
DO_PREPROCESS=${5:-"true"}
DO_PRETRAIN=${6:-"true"}
OTHER_ARGS=${7:-""}
LR=${8:-"5e-4"}

LOGDIR=logs/train
mkdir -p ${LOGDIR}

# Preprocessing
if [[ "${DO_PREPROCESS}" = "true" ]]; then
    echo "Starting preprocessing for task ${TASK}..."
    bash preprocess_general.sh "${TASK}" "${PRETRAIN_DATA_PATH}" "${CACHE}" "${OTHER_ARGS}" \
        > "${LOGDIR}/${TASK}_preprocess_${SUFFIX}.log" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "Error during preprocessing. Check log: ${LOGDIR}/${TASK}_preprocess_${SUFFIX}.log"
        exit 1
    fi
    echo "Preprocessing complete."
fi

# Pretraining
if [[ "${DO_PRETRAIN}" = "true" ]]; then
    echo "Starting pretraining for task ${TASK}..."
    bash pretrain_general.sh "${TASK}" "${PRETRAIN_DATA_PATH}" "0,1,2,3" 4 "${TASK}_${SUFFIX}" "${PORT}" "${PRETRAIN_OUTPUT_DIR}" "${CACHE}" "${OTHER_ARGS}" "${LR}" \
        > "${LOGDIR}/${TASK}_pretrain_${SUFFIX}.log" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "Error during pretraining. Check log: ${LOGDIR}/${TASK}_pretrain_${SUFFIX}.log"
        exit 1
    fi
    echo "Pretraining complete."
fi

echo "All tasks completed successfully."
