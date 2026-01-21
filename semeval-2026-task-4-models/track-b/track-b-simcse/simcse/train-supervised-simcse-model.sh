#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch \
    --nproc_per_node $NUM_GPU \
    --master_port $PORT_ID "./train.py" \
    --model_name_or_path "../checkpoints/princeton-nlp-sup-simcse-roberta-large" \
    --train_file "../data/train_track_a.csv" \
    --output_dir "../train/checkpoints/sup-simcse-roberta-large-v1" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 60 \
    --save_steps 60 \
    --pooler_type cls \
    --overwrite_output_dir \
    --do_mlm \
    --mlm_weight 0.1 \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --hard_negative_weight 0.1 \
    --logging_steps 10 \
    --logging_dir "../train/checkpoints/sup-simcse-roberta-large-v1/runs" \
#    --fp16 \
    "$@"
