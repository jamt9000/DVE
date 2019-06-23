#!/bin/bash

# There is no val set for LSMDC, so we fix the number of epochs to be able to
# compare across methods

set -x
set -e


LOG_NAME=$1
GPU=$2
WORKERS=$3

export PYTHONUNBUFFERED="True"
echo "Launching with ${WORKERS} on GPU: ${GPU}"
EXP_NAME="scarce-data"

CONFIG_DIR="data/gen_configs"
LOG="/scratch/shared/nfs1/albanie/exp/objectframe-pytorch/grid-logs/${LOG_NAME}-${EXP_NAME}-GPU${GPU}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
export PYTHONPATH="${HOME}/local/anaconda3/bin"
source activate pt37

num_devices=4
count=1
for config in "$CONFIG_DIR/${EXP_NAME}"/*
do
    if [[ $(( count % WORKERS )) == $GPU ]]; then
        echo "LAUNCHING: ${config}"
        python train.py --config ${config} --device $GPU
    fi
    count=$((count+1))
done
