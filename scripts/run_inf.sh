#! /bin/bash

GPU=0
DATA_DIR=./Data/dev
EXP_DIR=./Exps
COMMON_TASK_NAME=try
EVAL_BS=2
NUM_GPUS=1
MODEL_STR=GIT

echo "---> ${MODEL_STR} Run"
CUDA_VISIBLE_DEVICES=${GPU} ./scripts/inf.sh ${NUM_GPUS} \
    --data_dir ${DATA_DIR} --exp_dir ${EXP_DIR} --task_name ${COMMON_TASK_NAME} \
    --eval_batch_size ${EVAL_BS} \
    --cpt_file_name ${MODEL_STR} \
    --skip_train True \
    --re_eval_flag True \
    --eval_epoch 12
