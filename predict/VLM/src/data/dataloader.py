from predict.VLM.src.data.preprocessing import preprocess_messages

from copy import deepcopy
import torch
from qwen_vl_utils import process_vision_info


def get_collate_fn(processor, is_train=True):
    def train_collate_fn(batch):
        all_texts, all_images = [], []

        for sample in batch:
            messages = preprocess_messages(
                sample["messages"],
                processor,
                include_assistant=True,
                add_eos=True
            )

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            all_texts.append(text)

            image_inputs, _ = process_vision_info(messages)
            all_images.append(image_inputs)

        model_inputs = processor(
            text=all_texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        )

        # label masking 
        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone().fill_(-100)

        assistant_id = processor.tokenizer.convert_tokens_to_ids("<|assistant|>")
        assistant_id = torch.tensor(assistant_id, device=input_ids.device)

        for i in range(input_ids.size(0)):
            pos = torch.where(input_ids[i] == assistant_id)[0]
            if len(pos) > 0:
                labels[i, pos[-1] + 1 :] = input_ids[i, pos[-1] + 1 :]

        model_inputs["labels"] = labels
        return model_inputs
    
    
    def eval_collate_fn(batch):
        all_texts, all_images = [], []
        all_labels, all_prev_closes = [], []

        for sample in batch:
            messages = preprocess_messages(
                sample["messages"],
                processor,
                include_assistant=False
            )
            
            for msg in sample["messages"]:
                if msg["role"] == "assistant":
                    text_idxs = [i for i, c in enumerate(msg["content"]) if c["type"] == "text"]
                    if text_idxs:
                        labels = msg["content"][text_idxs[-1]]["text"]

            all_labels.append(labels)
            all_prev_closes.append(sample.get("prev_close", None))

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            all_texts.append(text)
            
            image_inputs, _ = process_vision_info(messages)
            all_images.append(image_inputs)

        model_inputs = processor(
            text=all_texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        )

        return {
            "model_inputs": model_inputs,
            "labels": all_labels,
            "prev_close": all_prev_closes
        }
        
        
    return train_collate_fn if is_train else eval_collate_fn