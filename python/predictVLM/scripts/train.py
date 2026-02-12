from predictVLM.src.configs.train_config import TrainConfig
from predictVLM.src.utils.set_seed import set_seed
from predictVLM.src.data.dataloader import get_collate_fn
from predictVLM.src.engine.trainer import validate

import os
import tyro
import torch
from torch.utils.data import DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    Qwen2_5_VLConfig.vocab_size = 152064
    model.config.vocab_size = 152064
    
    # OOM 방지용
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    min_pixels = 32*28*28
    max_pixels = 64*28*28
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True
    )
    
    # assistant 토큰 추가
    if "<|assistant|>" not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<|assistant|>"]})
    model.resize_token_embeddings(152064)
    setattr(Qwen2_5_VLConfig, "vocab_size", 152064)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 데이터 셋
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(cfg.data_dir, f"preprocessing/train.jsonl"),
            "valid": os.path.join(cfg.data_dir, f"preprocessing/val.jsonl"),
        }
    )
    
    training_args = TrainingArguments(
        output_dir=cfg.checkpoint_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=get_collate_fn(processor, is_train=True)
    )
    valid_loader = DataLoader(dataset["valid"], batch_size=1, collate_fn=get_collate_fn(processor, is_train=False))
    
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\n======== Epoch {epoch+1} ========")
    
        trainer.args.num_train_epochs = 1
        trainer.args.output_dir = os.path.join(cfg.checkpoint_dir, f"{epoch + 1}epoch")
        trainer.train()

        validate(model, processor, valid_loader, cfg.horizons, device=device)


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)