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
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import PeftModel
from datasets import load_dataset


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    if cfg.best_epoch == -1:
        print("best epoch 설정 필요")
        return
    else:
        model = PeftModel.from_pretrained(model, os.path.join(cfg.checkpoint_dir, f"{cfg.best_epoch}epoch"), torch_dtype=torch.float16)

    min_pixels = 32*28*28
    max_pixels = 64*28*28
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True
    )
    
    dataset = load_dataset(
        "json",
        data_files={
            "test": os.path.join(cfg.data_dir, f"preprocessing/test.jsonl"),
        }
    )
    test_loader = DataLoader(dataset["test"], batch_size=1, collate_fn=get_collate_fn(processor, is_train=False))
    
    validate(model, processor, test_loader, cfg.horizons, device=device)
    # 추후 사용 시, inference 코드 필요! 지금은 우선 test도 metrics 계산만


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)