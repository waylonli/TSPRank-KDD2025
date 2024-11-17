#!/bin/bash

MODE=$1
MARKET=$2

if [ "$MODE" == "train" ]; then
  python ./run_stock.py train \
    --market ${MARKET} \
    --sector all \
    --pipeline baseline \
    --epochs 100 \
    --eval_freq 5 \
    --lr 2e-4 \
    --weight_decay 1e-5 \
    --eval_metric kendall_tau \
    --optimizer adam
else
  python ./run_stock.py test --pipeline baseline --checkpoint checkpoints/baseline_stock --market ${MARKET} --sector all
fi