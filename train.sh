#!/bin/bash

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export OMP_NUM_THREADS=3
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=1

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 train.py \
    --model model-path \
    --output_dir output-dir \
    --language language \
    --epochs 100 \
    --learning_rate 1e-5 \
    --warmup_steps 200 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 8 \
    --eval_accumulation_steps 8 \
    --early_stop_epoch true \
    --fixed_loss_weight true \
    --fixed_asr_weight 0.5 \
    --frozen_encoder false \
    --ner_audio_root_path ner-audio-root \
    --sa_audio_root_path sa-audio-root \
    --ner_dataset ner-dataset \
    --sa_dataset sa-dataset \
    --gradient_checkpointing true \
    --fp16 true \
    --save_total_limit 2 \
    --dataloader_num_workers 3 \
    --use_deepspeed false