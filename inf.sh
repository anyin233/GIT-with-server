#! /bin/bash

NUM_GPUS=$1
shift

python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} run_inf_task.py $*
