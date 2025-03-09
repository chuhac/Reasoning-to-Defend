import logging
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, HfArgumentParser, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model  
import datasets
import torch

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pre-trained model or model identifier from huggingface.co/models"}
    )

    use_lora: bool = field(
        default=False, metadata={"help": "Use LoRA or not"}
    )

@dataclass
class DataArguments:
    dataset_path: str = field(
        metadata={"help": "Dataset name from the Hugging Face datasets library"}
    )
    
@dataclass
class LoraArguments:
    r: int = field(
        default=8, metadata={"help": "Rank parameter for LoRA"}
    )
    lora_alpha: float = field(
        default=8, metadata={"help": "Hyper Alpha for LoRA"}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None, 
    )
    target_modules: Optional[List[str]] = field(
        default=None, 
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout probability for Lora layers."}
    )
    bias: str = field(
        default="none", metadata={"help": "bias"}
    )


def compute_loss_func(outputs, labels, num_items_in_batch):
    logits = outputs.logits.float()
    vocab_size = logits.shape[-1]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="mean")

    safe_index = vocab_size - 8
    unsafe_index = vocab_size - 8 + 1
    rethink_index = vocab_size - 8 + 2

    indices_safe = (shift_labels == safe_index)
    indices_unsafe = (shift_labels == unsafe_index)

    if indices_safe.any():
        logits_diff_safe = shift_logits[indices_safe, safe_index] - shift_logits[indices_safe, unsafe_index]
        loss += -torch.nn.functional.logsigmoid(logits_diff_safe).mean()
    
    if indices_unsafe.any():
        logits_diff_unsafe = shift_logits[indices_unsafe, unsafe_index] - shift_logits[indices_unsafe, safe_index]
        loss += -torch.nn.functional.logsigmoid(logits_diff_unsafe).mean()

    return loss

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Training/evaluation parameters {training_args}")

    dataset = datasets.load_dataset(data_args.dataset_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    print(lora_args)
    lora_config = LoraConfig(
        **lora_args.__dict__
    )

    if model_args.use_lora:
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        

    def tokenize_function(example):
        inst_enc = tokenizer.encode(example["instruction"], padding=False, truncation=True)
        inp_enc = tokenizer.encode(example["input"], padding=False, truncation=True)
        out_enc = tokenizer.encode(example["output"], padding=False, truncation=True)
        
        input_ids = inst_enc + inp_enc + out_enc + [tokenizer.eos_token_id]
        labels = [-100] * len(inst_enc + inp_enc) + out_enc + [tokenizer.eos_token_id]

        return dict(input_ids=input_ids, labels=labels)


    tokenized_datasets = dataset.map(tokenize_function, batched=False, num_proc=32)
    train_dataset = tokenized_datasets["train"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        compute_loss_func=compute_loss_func
    )

    trainer.train()

    model_merged = model.merge_and_unload() if model_args.use_lora else model
    model_merged.save_pretrained(training_args.output_dir)

    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()