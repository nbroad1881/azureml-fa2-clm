"""
Basic training script for CLM with Flash Attention 2.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from accelerate import Accelerator


class FA2Trainer(Trainer):
    """
    Flash attention can only be used in fp16 or bf16 mode.
    This trainer ensures that evaluation and prediction are done in the correct dtype.
    """

    def __init__(self, *args, eval_dtype="bf16", **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dtype = eval_dtype

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        dtype = torch.bfloat16 if self.eval_dtype == "bf16" else torch.float16
        enabled = self.eval_dtype in {"bf16", "fp16"}
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )


@dataclass
class Config(TrainingArguments):

    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    max_seq_length: int = field(default=4096, metadata={"help": "Max length for text"})

    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA alpha (usually 2x rank)"}
    )

    num_proc: int = field(
        default=20,
        metadata={"help": "Number of processes to use for preprocessing dataset"},
    )

    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={
            "help": "Attention implementation to use. Choose from ['flash_attention_2', 'sdpa', 'eager']"
        },
    )


def main():
    parser = HfArgumentParser((Config,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": Accelerator().process_index},
        quantization_config=qconfig,
        attn_implementation=args.attn_implementation,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules="all-linear",
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()



    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

    # REPLACE THIS WITH YOUR OWN DATASET
    ds = load_dataset("ccdv/pubmed-summarization")

    # Reducing number of examples for faster training
    ds["train"] = ds["train"].select(range(10000))

    # REPLACE THIS WITH YOUR OWN FORMATTING FUNCTION
    def format_text(example):
        """
        This function formats the text to be used by the model.

        For summarization, this looks like:
            [Docs]

            Summary:
            [Summary]

        This script uses the `ccdv/pubmed-summarization` dataset from Hugging Face,
        where the model aims to summarize the article. The abstract is the ideal summary.

        Note that I am truncating the text from the article because some of them are extremely long.
        You should do EDA to determine the best truncation length for your dataset.
        """

        article = example["article"][:10000]
        abstract = example["abstract"][:2000]

        return {"text": f"{article}\n\nSummary:\n{abstract}"}

    def tokenize(batch):
        """
        Mistral tokenizer does not put eos token by default. We add it in this function before tokenizing.
        This ensures that the model will stop generating after the summary.

        There is the slight risk that truncation will cut off the summary and eos token,
        but this should be for a very small number of examples if the max_seq_length is set correctly.
        """

        tokenized = tokenizer(
            [x + tokenizer.eos_token for x in batch["text"]],
            padding=False,
            truncation=True,
            max_length=args.max_seq_length,
        )

        tokenized["labels"] = tokenized.input_ids.copy()

        return tokenized

    with args.main_process_first(desc="dataset map formatting and tokenization"):
        ds = ds.map(format_text, num_proc=args.num_proc, desc="Formatting text")

        keep_cols = ["input_ids", "attention_mask", "labels"]
        remove_cols = [x for x in ds["train"].column_names if x not in keep_cols]
        ds = ds.map(
            tokenize,
            batched=True,
            num_proc=args.num_proc,
            desc="Tokenizing",
            remove_columns=remove_cols,
        )

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=16, max_length=args.max_seq_length
    )

    # Specify dtype for evaluation
    if args.bf16:
        eval_dtype = "bf16"
    elif args.fp16:
        eval_dtype = "fp16"
    else:
        eval_dtype = "fp32"

    trainer = FA2Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dtype=eval_dtype,
    )

    if args.process_index == 0:
        # Show first example
        sample = data_collator([ds["train"][0]])

        labels = sample["labels"]
        labels = labels[labels != -100]
        print(labels)
        print(tokenizer.decode(labels.tolist()))
        print("last checkpoint:", args.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
