import os
from jailbreakbench.classifier import LlamaGuard1JailbreakJudgeLocal, LlamaGuard3JailbreakJudgeLocal
import argparse
from datasets import load_dataset, Dataset


def get_args():
    parser = argparse.ArgumentParser(description="Process some parameters for the AI model defense system.")

    parser.add_argument('--jailbreak_name', type=str, default="PAIR",
                        help='Name of the jailbreak method to use (default: PAIR)')
    parser.add_argument('--model_name', type=str, default="vicuna-13b-v1.5",
                        help='Name of the model to use (default: vicuna-13b-v1.5)')
    parser.add_argument('--defense_name', type=str, default="SynonymSubstitution",
                        help='Name of the defense technique to use (default: SynonymSubstitution)')
    parser.add_argument('--prefix_folder', type=str, )

    args = parser.parse_args()

    print(f"Jailbreak Name: {args.jailbreak_name}")
    print(f"Model Name: {args.model_name}")
    print(f"Defense Name: {args.defense_name}")

    return args


if __name__ == "__main__":
    args = get_args()
    jailbreak_name = args.jailbreak_name
    model_name = args.model_name
    defense_name = args.defense_name
    prefix_folder = args.prefix_folder
    jailbreak_judge = LlamaGuard3JailbreakJudgeLocal(api_key="", device="cuda:0")
    ds = load_dataset("json", data_files=f"{prefix_folder}/{model_name}-{jailbreak_name}-{defense_name}.jsonl", split="train")
    results = jailbreak_judge.classify_responses(ds["prompt"], ds["response"])
    
    ds = ds.to_dict()
    ds["v3_jailbreak_flag"] = results
    ds = Dataset.from_dict(ds)

    ds.to_json(f"{prefix_folder}/{model_name}-{jailbreak_name}-{defense_name}.jsonl")
    with open(f"{prefix_folder}/jbb_results.log", 'a') as file:
        print(f"{model_name}\t{jailbreak_name}\t{defense_name}\t{sum(list(map(int, results))) / ds.num_rows}", file=file)

