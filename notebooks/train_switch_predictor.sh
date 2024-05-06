# WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,3,6,7 torchrun train_pattern_predictor.py \

# WANDB_MODE=online CUDA_VISIBLE_DEVICES=1 \
# python -m ipdb \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 \
torchrun --nproc_per_node=1 --master_port=26718 \
train_switch_pattern_predictor.py \
   --model_name_or_path google-t5/t5-base \
   --run_name t5base_predictor_switch64_bigbench \
    --output_dir ./logs/ \
    --bf16 True \
    --tf32 True \
    --evaluation_strategy "epoch" \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 20 \
    --num_train_epochs 2 \
    --load_best_model_at_end True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 4e-5 \
    --weight_decay 5e-3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    $@