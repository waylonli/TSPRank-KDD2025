#!/bin/bash

MODE=$1
MARKET=$2

if [ "$MODE" == "train" ]; then
  python ./run_stock.py train \
    --market ${MARKET} \
    --sector all \
    --pipeline rankformer \
    --epochs 50 \
    --eval_freq 5 \
    --lr 2e-4 \
    --weight_decay 1e-5 \
    --tf_num_layers 1 \
    --dim_emb 128 \
    --tf_nhead 8 \
    --tf_dim_ff 128 \
    --batch_size 128 \
    --eval_metric kendall_tau \
    --optimizer adam
else
  python ./run_stock.py test --pipeline rankformer --checkpoint checkpoints/rankformer_stock_local --market ${MARKET} --sector all
fi