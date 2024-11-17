#!/bin/bash

MODE=$1
TOP=$2

if [ "$MODE" == "train" ]; then
  for FOLD in 1 2 3 4 5
  do
    python ./run_mq2008.py train \
      --dataset mq2008 \
      --pipeline rankformer \
      --epochs 100 \
      --eval_freq 5 \
      --tf_num_layers 4 \
      --tf_nhead 8 \
      --lr 2e-4 \
      --weight_decay 1e-5 \
      --tf_dim_ff 128 \
      --dim_emb 128 \
      --eval_metric kendall_tau \
      --batch_size 64 \
      --optimizer adam \
      --fold ${FOLD} \
      --top ${TOP}
  done
else
  for FOLD in 1 2 3 4 5
  do
    python ./run_mq2008.py test --pipeline rankformer --checkpoint checkpoints/rankformer_mq2008_fold${FOLD}_${TOP}
  done
fi