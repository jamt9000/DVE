#!/bin/bash

LOGNAME="`date +'%Y-%m-%d_%H-%M-%S'`"
WORKERS=4
for GPU in $(seq 0 $((WORKERS-1)))
do
    echo $GPU
    ./launcher.sh $LOGNAME $GPU $WORKERS &
done
