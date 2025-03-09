export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p ${OUTPUTS_PATH}/r2d_outputs

MODEL_NAME_LIST=("vicuna-13b-v1.5" "Meta-Llama-3-8B" "Mistral-7B-v0.3" "vicuna-7b-v1.5" "Qwen2-7B" "Qwen2.5-14B")

for MODEL_NAME in "${MODEL_NAME_LIST[@]}"; do
    echo ${MODEL_NAME}
    python expand.py --model_name ${MODEL_NAME}

    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 train.py \
        --model_name_or_path ${OUTPUTS_PATH}/r2d_outputs/${MODEL_NAME}-expanded \
        --dataset_path chuhac/R2D-R1 \
        --use_lora \
        --bf16 \
        --output_dir ${OUTPUTS_PATH}/r2d_outputs/${MODEL_NAME}-r2d \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-4 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.02 \
        --r 64 \
        --lora_alpha 64. \
        --report_to none \
        --gradient_checkpointing \
        --modules_to_save "embed_tokens" "lm_head" \
        --target_modules "q_proj" "k_proj" "v_proj" "up_proj" "down_proj" \
        --ddp_find_unused_parameters False \
        --logging_steps 10
done
