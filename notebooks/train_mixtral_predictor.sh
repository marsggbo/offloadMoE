
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 \
python train_mixtral_pattern_predictor.py \
   --model_name_or_path google-t5/t5-small \
   --output_dir ./logs/ \
   --bf16 True \
   --tf32 True \
   --eval_strategy "epoch" \
   --lazy_preprocess True \
   --save_strategy "epoch" \
   --save_steps 500 \
   --save_total_limit 1 \
   --logging_steps 20 \
   --num_train_epochs 50 \
   --load_best_model_at_end True \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --learning_rate 4e-5 \
   --weight_decay 0.01 \
   --warmup_ratio 0.1 \
   --max_grad_norm 0.8 \
   --lr_scheduler_type "cosine" \
   --lr_head 1e-3 \
   --lr_base 1e-3 \
   $@
# bash train_mixtral_predictor.sh --run_name t5base_predictor_switch128_bigbench32k_lrhead1e-3_lrbase5e-4_bs16 --lr_head 1e-3 --lr_base 5e-4
# bash train_mixtral_predictor.sh \
# --run_name t5base_predictor_switch128_bigbench_lrhead1e-3_lrbase5e-4_bs16 \
# --lr_head 1e-3 \
# --lr_base 5e-4 