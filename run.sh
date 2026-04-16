#!/usr/bin/env bash
set -euo pipefail

model_name=${1:-CAMEL}
data_root=${DATA_ROOT:-../../data/pems}
data_file=${DATA_FILE:-tfnsw.csv}
dataset_tag=${DATASET_TAG:-$(basename "${data_file}" .csv)}
gap_days=${GAP_DAYS:-"365 548 730"}
sample_rate=${SAMPLE_RATE:-0.1}

for pred_len in 96; do
  for gap_day in ${gap_days}; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --train_seed 2024 \
      --sample_seed 7 \
      --samle_rate "${sample_rate}" \
      --gap_day "${gap_day}" \
      --root_path "${data_root}" \
      --data_path "${data_file}" \
      --model_id "${dataset_tag}_all_96_${pred_len}_${gap_day}" \
      --model "${model_name}" \
      --data custom \
      --target '' \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "${pred_len}" \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 27 \
      --dec_in 27 \
      --c_out 27 \
      --des CAMEL \
      --itr 1 \
      --learning_rate 0.0005 \
      --train_epochs 200 \
      --patience 5 \
      --lradj type3 \
      --camel_gap_years 1.5 \
      --camel_k_retrieve 8 \
      --camel_latent_dim 32 \
      --camel_d_model 32 \
      --lambda_mem 0.0 \
      --lambda_ode 0.01 \
      --lambda_smooth 0.0 \
      --steps_per_day 24
  done
done
