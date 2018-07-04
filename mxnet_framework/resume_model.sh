#!/bin/bash

# set -x
set -e

export PYTHONUNBUFFERED="True"
EXP_NAME=$1

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--exp)
    EXP_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    -l|--log)
    LOG_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--epoch)
    LAST_EPOCH="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


BASE_DIR="exps/${EXP_NAME}"

if ! [ -d "$BASE_DIR" ]; then
    exit 0
fi

# create log directory
LOG_DIR="${BASE_DIR}/logs"
mkdir -p $LOG_DIR

# create output directory
OUT_DIR="${BASE_DIR}/output"
mkdir -p $OUT_DIR

if [ -z "${LOG_NAME}" ]; then
    echo "USING NEW LOG FILE"
    LOG="${BASE_DIR}/logs/`date +'%Y-%m-%d_%H-%M-%S'`_train.txt"
else
    LOG=`find ${BASE_DIR}/logs -type f -ipath "*${LOG_NAME}*train.txt" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
fi

LAST_EPOCH=`find ${BASE_DIR}/snapshots -type f -ipath "*${LAST_EPOCH}*.params" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" " | xargs basename | tr -cd '[[:digit:]]'`

echo EXP_NAME = ${EXP_NAME}
echo LOG_NAME = ${LOG_NAME}
echo LOG = ${LOG}
echo LAST_EPOCH = ${LAST_EPOCH}

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python3 solve.py --exp-name ${EXP_NAME} --resume_model ${LAST_EPOCH}

HTML_OUTPUT="${BASE_DIR}/output/train_loss_`date +'%Y-%m-%d_%H-%M-%S'`.html"
python3 plot_curve.py --log ${LOG} --output ${HTML_OUTPUT}