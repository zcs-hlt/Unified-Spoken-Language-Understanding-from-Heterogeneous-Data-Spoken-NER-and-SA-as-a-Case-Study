#!/bin/bash

CHECKPOINT=${1:-"checkpoint-path"}
PROCESSOR_PATH="processor-path"
OUTPUT_DIR="output-dir"

export CUDA_VISIBLE_DEVICES=0

python inference.py \
    --model_path $CHECKPOINT \
    --processor_path $PROCESSOR_PATH \
    --language en \
    --ner_data_path none \
    --ner_audio_root none \
    --ner_dataset slue \
    --sa_data_path sa-data-path \
    --sa_audio_root sa-audio-root \
    --sa_dataset slue \
    --batch_size 48 \
    --beam_size 1 \
    --output_dir $OUTPUT_DIR

