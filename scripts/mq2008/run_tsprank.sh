#!/bin/bash

MODE=$1
LEARN=$2
TOP=$3

if [ "$MODE" == "train" ]; then
  for FOLD in 1 2 3 4 5
  do
    python ./run_mq2008.py train \
      --dataset mq2008 \
      --pipeline marginal \
      --epochs 100 \
      --eval_freq 5 \
      --tf_num_layers 4 \
      --tf_nhead 8 \
      --lr 1e-4 \
      --weight_decay 1e-5 \
      --tf_dim_ff 128 \
      --dim_emb 128 \
      --eval_metric kendall_tau \
      --learning ${LEARN} \
      --tsp_solver gurobi \
      --batch_size 64 \
      --optimizer adam \
      --fold ${FOLD} \
      --top ${TOP}
  done
else
  for FOLD in 1 2 3 4 5
  do
    python ./run_mq2008.py test --pipeline marginal --checkpoint checkpoints/marginal_mq2008_fold${FOLD}_${TOP}_${LEARN} --tsp_solver gurobi
  done
fi