#!/bin/bash

MODE=$1
LEARN=$2
TOP=$3

if [ "$MODE" == "train" ]; then
  for FOLD in 1 2 3 4 5
  do
    python ./run_event.py train \
    --dataset otd2  \
    --pipeline marginal  \
    --epochs 10  \
    --eval_freq 1  \
    --tf_num_layers 0  \
    --tf_nhead 4  \
    --embed openai  \
    --lr 1e-4  \
    --weight_decay 1e-5  \
    --tf_dim_ff 512  \
    --eval_metric kendall_tau  \
    --learning ${LEARN}  \
    --tsp_solver gurobi  \
    --batch_size 32  \
    --optimizer adam  \
    --top ${TOP}  \
    --fold ${FOLD}
  done
else
  for FOLD in 1 2 3 4 5
  do
    python ./run_event.py test --pipeline marginal --checkpoint checkpoints/marginal_otd2_fold${FOLD}_${TOP}_${LEARN} --tsp_solver gurobi
  done
fi


