import argparse
from dataclasses import dataclass, fields, asdict
from typing import Optional
from utils import get_logger

logger = get_logger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@dataclass
class TrainingConfig:
    model: str = "model-path"
    language: str = "en"
    task: str = "transcribe"
    output_dir: str = "output-dir"
    epochs: int = 100
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    eval_accumulation_steps: int = 16
    early_stop_epoch: bool = True
    fixed_loss_weight: bool = False
    fixed_asr_weight: float = 0.5
    frozen_encoder: bool = False
    split_token: bool = True
    asr_only: bool = False
    ner_audio_root_path: str = "ner-audio-root"
    sa_audio_root_path: str = "sa-audio-root"
    ner_dataset: str = "ner-dataset"
    sa_dataset: str = "sa-dataset"
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    save_total_limit: int = 2
    local_rank: int = -1
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 2
    torch_compile: bool = False
    cache_features: bool = False
    use_deepspeed: bool = False
    deepspeed_stage: int = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model-path")
    parser.add_argument("--output_dir", type=str, default="output-dir")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_accumulation_steps", type=int, default=16)
    parser.add_argument("--early_stop_epoch", type=str2bool, default=True)
    parser.add_argument("--fixed_loss_weight", type=str2bool, default=False)
    parser.add_argument("--fixed_asr_weight", type=float, default=0.5)
    parser.add_argument("--frozen_encoder", type=str2bool, default=False)
    parser.add_argument("--ner_audio_root_path", type=str, default="ner-audio-root")
    parser.add_argument("--sa_audio_root_path", type=str, default="sa-audio-root")
    parser.add_argument("--ner_dataset", type=str, default="ner-dataset")
    parser.add_argument("--sa_dataset", type=str, default="sa-dataset")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--bf16", type=str2bool, default=False)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    parser.add_argument("--torch_compile", type=str2bool, default=False)
    parser.add_argument("--cache_features", type=str2bool, default=False)
    parser.add_argument("--use_deepspeed", type=str2bool, default=False)
    parser.add_argument("--deepspeed_stage", type=int, default=2)
    return parser.parse_args()


def create_config_from_args(args) -> TrainingConfig:
    return TrainingConfig(
        model=args.model,
        output_dir=args.output_dir,
        language=args.language,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        early_stop_epoch=args.early_stop_epoch,
        fixed_loss_weight=args.fixed_loss_weight,
        fixed_asr_weight=args.fixed_asr_weight,
        frozen_encoder=args.frozen_encoder,
        ner_audio_root_path=args.ner_audio_root_path,
        sa_audio_root_path=args.sa_audio_root_path,
        ner_dataset=args.ner_dataset,
        sa_dataset=args.sa_dataset,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        save_total_limit=args.save_total_limit,
        local_rank=getattr(args, 'local_rank', -1),
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        torch_compile=args.torch_compile,
        cache_features=args.cache_features,
        use_deepspeed=args.use_deepspeed,
        deepspeed_stage=args.deepspeed_stage,
    ) 