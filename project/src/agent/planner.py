from .prompt import messages_for
from .validator import decode_action


def next_action(qwen, transcript, route, sample, tools, observations, seconds):
    messages = messages_for(transcript, route, sample.context, sample.history, tools, observations)
    return decode_action(qwen.generate(messages, seconds), tools)
