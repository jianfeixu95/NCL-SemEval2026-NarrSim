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
    --run_tag "finetune-princeton-nlp-sup-simcse-roberta-large" \
    --model_name_or_path "../checkpoints/princeton-nlp-sup-simcse-roberta-large" \
    --train_file "../data/dev_track_a_train_augmented_shuffled.csv" \
    --eval_file "../data/dev_track_a_test.csv" \
    --output_dir "../runs" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --max_seq_length 128 \
    --evaluation_strategy steps \
    --metric_for_best_model eval_accuracy \
    --load_best_model_at_end \
    --eval_steps 5 \
    --save_steps 5 \
    --pooler_type cls \
    --overwrite_output_dir \
    --do_mlm \
    --mlm_weight 0.1 \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --hard_negative_weight 0.01 \
    --logging_steps 1 \
    --logging_dir "../logs" \
    --fp16 \
    "$@"
