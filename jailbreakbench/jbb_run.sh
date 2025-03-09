pip install -e .
export INPUTS_PATH=/path/to/inputs
export OUTPUTS_PATH=/path/to/outputs
export CODE_PATH=/path/to/code/Reasoning-to-Defend
export HF_TRUST_REMOTE_CODE="1"
mkdir -p $OUTPUTS_PATH/jbb_outputs

CUDA_VISIBLE_DEVICES="6,7" python jbb_response.py --jailbreak_name JBC --model_name llama-3-8b --defense_name None --prefix_folder ${OUTPUTS_PATH}/jbb_outputs
CUDA_VISIBLE_DEVICES="6,7" python jbb_classify.py --jailbreak_name JBC --model_name llama-3-8b --defense_name None --prefix_folder ${OUTPUTS_PATH}/jbb_outputs

