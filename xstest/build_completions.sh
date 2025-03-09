mkdir -p ./model_competions
model_names=("Meta-Llama-3-8B" "Mistral-7B-v0.3" "Qwen2.5-14B" "Qwen2-7B" "vicuna-13b-v1.5" "vicuna-7b-v1.5")

for model_name in "${model_names[@]}"; do
    python build_completions.py --model_name_or_path ${INPUTS_PATH}/pretrained_models/${model_name}
    ray stop
done

