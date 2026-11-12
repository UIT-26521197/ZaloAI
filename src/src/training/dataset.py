from .loss import completion_labels


class CompletionDataset:
    def __init__(self, examples, tokenizer, context_tokens, new_tokens):
        self.rows = []
        self.skipped = 0
        for prompt, completion in examples:
            prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            answer = tokenizer(
                completion, add_special_tokens=False,
            )["input_ids"] + [tokenizer.eos_token_id]
            if len(prefix) + new_tokens > context_tokens or len(answer) > new_tokens:
                self.skipped += 1
                continue
            ids = prefix + answer
            self.rows.append({"input_ids": ids, "labels": completion_labels(ids, len(prefix))})
        if not self.rows:
            raise ValueError("No training/evaluation examples fit the token budgets")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]
