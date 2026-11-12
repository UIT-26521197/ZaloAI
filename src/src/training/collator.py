from .loss import IGNORE_INDEX


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch

        size = max(len(row["input_ids"]) for row in rows)
        input_ids, labels, attention_mask = [], [], []
        for row in rows:
            length = len(row["input_ids"])
            pad = size - length
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad)
            labels.append(row["labels"] + [IGNORE_INDEX] * pad)
            attention_mask.append([1] * length + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }
