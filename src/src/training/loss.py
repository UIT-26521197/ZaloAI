IGNORE_INDEX = -100


def completion_labels(token_ids, prompt_length):
    return [IGNORE_INDEX] * prompt_length + token_ids[prompt_length:]
