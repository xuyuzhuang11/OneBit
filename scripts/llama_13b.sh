#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=g0001,g0002
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --output=output_llama_13b.txt


MODEL_NAME=llama_13b
SCRIPT_DIR=$(dirname "$0")
DATASET=kd_132k
TEACHER_MODEL_PATH=/home/xyz/models/llama-13b
STUDENT_MODEL_PATH=$SCRIPT_DIR/ckpt_$MODEL_NAME
STUDENT_MODEL_START_PATH=$SCRIPT_DIR/start_$MODEL_NAME
STUDENT_MODEL_EXIST_PATH=$SCRIPT_DIR/ckpt_$MODEL_NAME/checkpoint-5000


python $SCRIPT_DIR/build_start_ckpt.py $MODEL_NAME

rm -rf $STUDENT_MODEL_PATH
cd $SCRIPT_DIR/../llama_factory || exit

deepspeed --num_nodes 2 --master_port=9901 --hostfile="$SCRIPT_DIR/hostfile" train_bash.py \
    --deepspeed "$SCRIPT_DIR/ds_config.json" \
    --stage kd \
    --kd_alpha 1.0 \
    --kd_beta 1 \
    --kd_loss_scale 0.01 \
    --cutoff_len 2048 \
    --model_name_or_path $STUDENT_MODEL_START_PATH \
    --teacher_model_name_or_path $TEACHER_MODEL_PATH \
    --do_train \
    --dataset $DATASET \
    --dataset_dir $SCRIPT_DIR/../data \
    --template vanilla \
    --finetuning_type full \
    --output_dir $STUDENT_MODEL_PATH \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine --warmup_steps 500 \
    --adam_beta1 0.9 --adam_beta2 0.98 --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 5000 \
    --learning_rate 2e-4 \
    --num_train_epochs 50.0 \
    --plot_loss \
    --fp16

    # --resume_from_checkpoint $STUDENT_MODEL_EXIST_PATH \
