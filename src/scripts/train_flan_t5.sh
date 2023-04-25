#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
output_path=/workspace/output/flan-t5-large-ABSA-rest-all-annotion-15epoch

# 3090 * 8 on t5-700M
CUDA_VISIBLE_DEVICES=2,4,6,7 deepspeed --master_port $port ../src/run_uie_ABSA.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path /workspace/MODELS/flan-t5-large \
   --data_dir /workspace \
   --task_config_dir /workspace/new_input_type/configs/ABSA_configs \
   --output_dir $output_path \
   --instruction_file /workspace/new_input_type/configs/instruction_config.json \
   --instruction_strategy single \
   --input_record_file $output_path/flan-t5-large.record \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --num_train_epochs 15 \
   --deepspeed /workspace/new_input_type/configs/ds_configs/stage0.config \
   --run_name flan-t5-large-ABSA-experiment \
   --max_source_length 512 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --max_num_instances_per_task 10000 \
   --max_num_instances_per_eval_task 200 \
   --add_dataset_name False \
   --add_task_name False \
   --num_examples 0 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 100 \
   --evaluation_strategy no \
   --save_strategy steps \
   --save_steps 2000 \
   --test_format test \
   --over_sampling 0
