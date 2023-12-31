#!/bin/bash

data_path="./task_data/mixtasks1-5_train.tsv"
model_name="codellama7b"
output_dir="./instruct_logging/codellama7b_task1-5"

nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 train.py \
    --deepspeed ds_config.json \
    --model_name_or_path $model_name \
    --data_path $data_path \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --model_max_length 1024 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1000 \
    --save_total_limit 15 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to none \
    --bf16 True \
    > train.log
