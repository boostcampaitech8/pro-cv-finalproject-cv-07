from predict.VLM.src.metrics.metrics import compute_regression_metrics

import ast
import torch
import json
from tqdm import tqdm


def validate(model, processor, val_loader, horizons, device="cuda"):
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    all_prev_close = []
    
    keys = [f"t+{i}" if i != 0 else "t" for i in range(horizons)]

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", total=len(val_loader)):
            inputs = batch["model_inputs"].to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=350, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            label_dicts = [ast.literal_eval(lbl) for lbl in batch["labels"]]   # [json.loads(lbl) for lbl in batch["labels"]]
            output_dicts = [ast.literal_eval(txt) for txt in output_texts]

            y_true = torch.tensor([[lbl[k] for k in keys] for lbl in label_dicts], dtype=torch.float32)
            y_pred = torch.tensor([[out[k] for k in keys] for out in output_dicts], dtype=torch.float32)
            prev_close = torch.tensor(batch["prev_close"], dtype=torch.float32)

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_prev_close.append(prev_close)

    all_y_true = torch.cat(all_y_true, dim=0) 
    all_y_pred = torch.cat(all_y_pred, dim=0)
    all_prev_close = torch.cat(all_prev_close, dim=0)

    compute_regression_metrics(
        y_true=all_y_true.cpu().numpy(),
        y_pred=all_y_pred.cpu().numpy(),
        horizons=keys,
        prev_close=all_prev_close.cpu().numpy(),
        verbose=True
    )