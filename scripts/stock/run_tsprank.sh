#!/bin/bash

MODE=$1
LEARN=$2
MARKET=$3

if [ "$MODE" == "train" ]; then
  python ./run_stock.py train \
    --market ${MARKET} \
    --sector all \
    --pipeline marginal \
    --epochs 150 \
    --eval_freq 5 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --tf_num_layers 1 \
    --dim_emb 128 \
    --tf_nhead 8 \
    --tf_dim_ff 128 \
    --batch_size 128 \
    --eval_metric kendall_tau \
    --tsp_solver gurobi \
    --optimizer adam \
    --learning ${LEARN}
else
  python ./run_stock.py test --pipeline marginal --checkpoint checkpoints/marginal_stock_${LEARN} --tsp_solver gurobi --market ${MARKET} --sector all
fi