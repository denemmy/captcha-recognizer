#!/bin/bash

# set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1
NET_FINAL=$2

BASE_DIR="exps/${EXP_NAME}"

if ! [ -d "$BASE_DIR" ]; then
    echo "No directory ${BASE_DIR}"
    exit 0
fi

# create log directory
LOG_DIR="${BASE_DIR}/logs"
mkdir -p $LOG_DIR

LOG="${BASE_DIR}/logs/`date +'%Y-%m-%d_%H-%M-%S'`_test.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ -n "$NET_FINAL" ];
then
   NET_FINAL=`find ${BASE_DIR}/snapshots -type f -ipath "*${NET_FINAL}*.params" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
else
   NET_FINAL=`find ${BASE_DIR}/snapshots -type f -ipath '*.params' -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
fi

echo ${NET_FINAL}
python3 test.py --exp-name ${EXP_NAME} --model ${NET_FINAL} --gpus 0,1,2,3

