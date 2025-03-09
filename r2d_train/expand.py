from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Process some parameters for the AI model defense system.")
    parser.add_argument('--model_name', type=str, default="vicuna-13b-v1.5",
                        help='Name of the model to use (default: vicuna-13b-v1.5)')
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()

    model_name = args.model_name
    model_path = os.path.join(os.environ["INPUTS_PATH"], "pretrained_models", model_name)

    llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["[SAFE]", "[UNSAFE]", "[RETHINK]", "<think>", "</think>", "[R2D-Reserve-0]", "[R2D-Reserve-1]", "[R2D-Reserve-2]"])

    llm.resize_token_embeddings(len(tokenizer)) 
    print(llm.model.embed_tokens.weight[-8:, :])
    print(llm.lm_head.weight[:, -8:])


    llm.config.vocab_size = len(tokenizer)
    llm.save_pretrained(os.path.join(os.environ["OUTPUTS_PATH"], "r2d_outputs", model_name + "-expanded"))
    tokenizer.save_pretrained(os.path.join(os.environ["OUTPUTS_PATH"], "r2d_outputs", model_name + "-expanded"))
