#!/bin/bash

# set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1

BASE_DIR="exps/${EXP_NAME}"

if ! [ -d "$BASE_DIR" ]; then
    exit 0
fi

LOG=`find ${BASE_DIR}/logs -type f -ipath '*train.txt' -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2 -d" "`
echo ${LOG}

DATE_STR="`date +'%Y-%m-%d_%H-%M-%S'`"

HTML_OUTPUT="${BASE_DIR}/output/train_loss_${DATE_STR}.html"

python3 plot_curve.py --log ${LOG} --output ${HTML_OUTPUT}