import logging

from ..config import load_config
from ..core.registry import ToolRegistry
from ..environment.adapter import catalog_path
from ..environment.initializer import load_environment
from ..models.laya import LayaModel
from ..models.loader import initialize_runtime, model_source
from ..models.qwen import load_qwen
from ..models.whisper import WhisperModel
from ..utils.logging import configure_logging
from .checkpoint import new_run_directory
from .collator import CompletionCollator
from .data import partition
from .dataset import CompletionDataset
from .evaluate import evaluate
from .preprocessing import examples

LOGGER = logging.getLogger(__name__)


def train(config):
    initialize_runtime(config)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    rows, held_out = partition(config)
    registry = ToolRegistry.load(catalog_path(config.data_root, "train"))
    open_sample = load_environment(config.data_root, "train")
    qwen_source = model_source(config.qwen, config.offline)
    tokenizer = AutoTokenizer.from_pretrained(qwen_source, local_files_only=config.offline)
    whisper = WhisperModel(model_source(config.whisper, config.offline), config.device, config.offline)
    laya = LayaModel(model_source(config.laya, config.offline), config.laya_device)

    def build(split):
        return CompletionDataset(
            examples(split, config, registry, open_sample, whisper, laya), tokenizer,
            config.context_tokens, config.new_tokens,
        )

    training, validation = build(rows), build(held_out)
    del whisper, laya
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _, model = load_qwen(qwen_source, config.device, config.offline)
    dtype = model.dtype
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    directory = new_run_directory(config.output)
    settings = TrainingArguments(
        output_dir=str(directory / "checkpoints"),
        seed=config.seed,
        data_seed=config.seed,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        num_train_epochs=config.epochs,
        use_cpu=model.device.type == "cpu",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        label_names=["labels"],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=settings, train_dataset=training, eval_dataset=validation,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    adapter = directory / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    evaluate(trainer, directory, training, validation, [row["id"] for row in held_out])
    LOGGER.info("Qwen adapter: %s", adapter)
    return adapter


def main(argv=None):
    configure_logging()
    return train(load_config("training", argv))


if __name__ == "__main__":
    main()
