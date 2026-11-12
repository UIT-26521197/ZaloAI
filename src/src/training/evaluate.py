import json

from ..utils.paths import atomic_text


def evaluate(trainer, directory, train, validation, validation_ids):
    metrics = trainer.evaluate()
    metrics.update({
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_examples_over_budget": train.skipped,
        "validation_examples_over_budget": validation.skipped,
    })
    with atomic_text(directory / "validation_metrics.json") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")
    with atomic_text(directory / "validation_ids.json") as stream:
        json.dump(validation_ids, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
