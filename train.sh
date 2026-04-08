#!/bin/bash
CUDA_VISIBLE_DEVICES=0 /path/to/python_env/bin/python train_end_to_end.py \
    --data_path /path/to/dataset \
    --log_dir /path/to/log_dir \
    --num_epochs 40 \
    --learning_rate 1e-4 \
    --scheduler_step_size 20 \
    --lora_rank 16 \
    --lora_alpha 1.0 \
    --lora_dropout 0.1 \
    --disparity_smoothness 1e-4 \
    --position_smoothness 0.001 \
    --transform_constraint 0.01 \
    --transform_smoothness 0.01 \
    --consistency_constraint 0.01 \
    --geometry_constraint 0.01 \
    --save_frequency 2 \
    --load_weights_folder /path/to/stage_one_model \
    --models_to_load position_encoder position \
    --pretrained_path /path/to/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
