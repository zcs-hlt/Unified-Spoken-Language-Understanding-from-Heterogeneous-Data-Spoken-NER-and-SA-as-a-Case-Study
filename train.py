import os
from transformers import WhisperProcessor, Seq2SeqTrainingArguments
from whisper.normalizers.english import EnglishTextNormalizer

from config import parse_args, create_config_from_args
from data_loader import load_datasets, DataCollatorSpeechSeq2SeqWithPadding
from model import create_model
from trainer import WhisperSeq2SeqTrainer
from utils import setup_logging, get_logger, log_config

logger = get_logger(__name__)


def get_deepspeed_config(config):
    if not config.use_deepspeed:
        return None
    
    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": "auto"
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto",
                "torch_adam": True,
                "adam_w_mode": True
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "communication_data_type": "fp16",
        "prescale_gradients": False,
        "gradient_predivide_factor": 1.0,
        "wall_clock_breakdown": False,
    }
    
    if config.deepspeed_stage == 1:
        ds_config["zero_optimization"] = {
            "stage": 1,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        }
    elif config.deepspeed_stage == 2:
        ds_config["zero_optimization"] = {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        }
    elif config.deepspeed_stage == 3:
        ds_config["zero_optimization"] = {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_gather_16bit_weights_on_model_save": True,
        }
    
    return ds_config


def setup_processor(config):
    if config.asr_only:
        processor = WhisperProcessor.from_pretrained(config.model)
    else:
        processor = WhisperProcessor.from_pretrained(
            'processor/whisper-small-Special', 
            language=config.language, 
            task=config.task
        )
    return processor


def create_training_args(config):
    ds_config = get_deepspeed_config(config)
    
    return Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        eval_accumulation_steps=config.eval_accumulation_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        num_train_epochs=config.epochs,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        bf16=config.bf16,
        eval_strategy="epoch",
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        overwrite_output_dir=True,
        report_to=["tensorboard"],
        logging_dir=f"{config.output_dir}/logs",
        dataloader_pin_memory=True,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        remove_unused_columns=False,
        load_best_model_at_end=not config.use_deepspeed,
        greater_is_better=False,
        prediction_loss_only=False,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if not config.use_deepspeed else None,
        local_rank=getattr(config, 'local_rank', -1),
        ddp_broadcast_buffers=False,
        torch_compile=config.torch_compile,
        deepspeed=ds_config,
    )


def main():
    global logger
    
    args = parse_args()
    config = create_config_from_args(args)
    
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        log_file = os.path.join(config.output_dir, "training.log")
        setup_logging(log_file=log_file)
        logger = get_logger(__name__)
    else:
        setup_logging()
    
    log_config(config, logger)
    processor = setup_processor(config)
    model = create_model(config, processor)
    training_args = create_training_args(config)
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    train_dataset, dev_dataset = load_datasets(config, processor)
    
    trainer = WhisperSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        processor=processor,
        early_stop_epoch=config.early_stop_epoch,
        fixed_loss_weight=config.fixed_loss_weight,
        fixed_asr_weight=config.fixed_asr_weight,
    )
    
    trainer.train()


if __name__ == '__main__':
    main() 