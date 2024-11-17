#!/bin/bash

MODE=$1
TOP=$2

if [ "$MODE" == "train" ]; then
  for FOLD in 1 2 3 4 5
  do
    python ./run_event.py train \
    --dataset otd2 \
    --pipeline rankformer \
    --epochs 20 \
    --fold ${FOLD} \
    --eval_freq 1 \
    --tf_num_layers 0 \
    --tf_nhead 4  \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --tf_dim_ff 128 \
    --embed openai \
    --eval_metric kendall_tau \
    --batch_size 64 \
    --optimizer adam \
    --top ${TOP}
  done
else
  for FOLD in 1 2 3 4 5
  do
    python ./run_event.py test --pipeline rankformer --checkpoint checkpoints/rankformer_otd2_fold${FOLD}_${TOP}
  done
fi