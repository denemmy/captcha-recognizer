#!/bin/bash

# set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1

BASE_DIR="exps/${EXP_NAME}"

# create log directory
LOG_DIR="${BASE_DIR}/logs"
mkdir -p $LOG_DIR

# create output directory
OUT_DIR="${BASE_DIR}/output"
mkdir -p $OUT_DIR

LOG="${BASE_DIR}/logs/`date +'%Y-%m-%d_%H-%M-%S'`_train.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python3 solve.py --exp-name ${EXP_NAME}

HTML_OUTPUT="${BASE_DIR}/output/train_loss_`date +'%Y-%m-%d_%H-%M-%S'`.html"
python3 plot_curve.py --log ${LOG} --output ${HTML_OUTPUT}