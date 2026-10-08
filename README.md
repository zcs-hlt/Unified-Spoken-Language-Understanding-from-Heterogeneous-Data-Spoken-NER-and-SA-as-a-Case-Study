# Unified Spoken Language Understanding from Heterogeneous Data: Spoken NER and SA as a Case Study

![Venue](https://img.shields.io/badge/NLPCC_2026-Accepted-4b8bbe.svg)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv%3A2507.12951-b31b1b.svg)](https://arxiv.org/abs/2507.12951)

> **Accepted at NLPCC 2026.**

**Zhichao Sheng, Shilin Zhou, Chen Gong, Zhenghua Li**

Institute of Artificial Intelligence, School of Computer Science and Technology, Soochow University

This repository contains the Whisper-based training and inference implementation for our paper. We jointly model **automatic speech recognition (ASR), spoken named entity recognition (NER), and spoken sentiment analysis (SA)** using heterogeneous datasets in a single generative framework. Each training example needs a transcript and the annotation for its own understanding task; aligned NER and sentiment annotations for every utterance are not required.

The [earlier preprint](https://arxiv.org/abs/2507.12951) is titled *UniSLU: Unified Spoken Language Understanding from Heterogeneous Cross-Task Datasets*. The title and results below follow the NLPCC camera-ready paper.

## Overview

Our approach combines three components:

- **Unified output representation.** Generate the transcript, a separator, a task-control token, and the corresponding task output in one sequence.
- **Shared generative model.** Fully fine-tune a Whisper encoder and decoder on spoken NER and SA data together, allowing both tasks to benefit from shared representations.
- **Dynamic weighted loss.** Balance transcription and understanding losses according to their token lengths, giving shorter task outputs greater weight.

The implementation uses the following output format:

```text
[ASR transcript][T/L][task-control token][task-specific output]

NER: this council made great progress[T/L][NER][ORG]council[/ORG]
SA:  this council made great progress[T/L][SA][S]Positive[/S]
```

`[T/L]` separates transcription from understanding. `[NER]` and `[SA]` select the task. English NER uses seven entity types: `PLACE`, `QUANT`, `ORG`, `WHEN`, `NORP`, `PERSON`, and `LAW`. Sentiment labels are wrapped in `[S]...[/S]` in the code.

For ASR and task token lengths, the loss weights are:

```text
W_ASR  = Len_task / (Len_ASR + Len_task)
W_task = Len_ASR  / (Len_ASR + Len_task)
L      = W_ASR * L_ASR + W_task * L_task
```

See [trainer.py](trainer.py) for the per-example weighting and token-weighted batch reduction.

## Main Results

Results reported in the NLPCC camera-ready paper on the **SLUE test sets**. VP and VC denote SLUE-VoxPopuli and SLUE-VoxCeleb. WER is lower-is-better; F1 and SLUE score are higher-is-better. All scores are percentages.

| Model | ASR WER (VP) ↓ | ASR WER (VC) ↓ | NER Micro-F1 ↑ | SA Macro-F1 ↑ | SLUE score ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ours, separate Whisper-medium models | 8.63 | 10.92 | 63.83 | 60.92 | 71.65 |
| Whisper-medium + Qwen2.5-7B, pipeline | 7.70 | 13.33 | 55.16 | 68.65 | 71.09 |
| Whisper-medium + Qwen2.5-7B, end-to-end | 11.63 | 14.16 | 64.05 | 63.27 | 71.47 |
| **Ours, unified Whisper-medium** | **8.77 ± 0.42** | **11.47 ± 1.40** | **69.95 ± 1.10** | **61.60 ± 1.76** | **73.81 ± 0.24** |

The unified model's results are averaged over three random seeds, with standard deviations. It improves the SLUE score by **2.34 points** over the end-to-end Whisper-medium + Qwen2.5-7B baseline. In the paper's efficiency experiment on one A100-40GB GPU with batch size 32, throughput increases from 3.89 to 19.35 samples/s, approximately **5×**.

The paper also reports Chinese experiments on CNERTA and CH-SIMS v2.0, achieving **60.62 NER F1** and **70.01 SA accuracy**, respectively. These are paper results, not measurements produced by the setup examples below.

## Repository Layout

```text
.
├── config.py          # Training configuration and command-line arguments
├── data_loader.py     # English/Chinese datasets, feature caching, padding
├── model.py           # Whisper initialization and custom dynamic decoder
├── trainer.py         # ASR/task loss splitting and dynamic weighting
├── train.py           # Training entry point and optional DeepSpeed setup
├── train.sh           # Example training launcher (edit placeholders first)
├── inference.py       # ASR/NER/SA evaluation and prediction export
├── inference.sh       # Example inference launcher (edit placeholders first)
├── mask.py            # Masking and beam-search utilities
├── utils.py           # Logging utilities
└── requirements.txt   # Python dependencies
```

Datasets, pretrained model weights, fine-tuned checkpoints, and the task-specific processor are **not bundled**. The instructions below describe preparation for a new training run. The LLM baselines in the paper are not implemented in this repository.

## Getting Started

### 1. Environment

Use Python 3.10+ and a CUDA-capable PyTorch environment for the training and FP16 inference commands below. Install matching `torch` and `torchaudio` builds for your CUDA version.

```bash
git clone https://github.com/zcs-hlt/Unified-Spoken-Language-Understanding-from-Heterogeneous-Data-Spoken-NER-and-SA-as-a-Case-Study.git
cd Unified-Spoken-Language-Understanding-from-Heterogeneous-Data-Spoken-NER-and-SA-as-a-Case-Study

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt "transformers==4.45.1" "tokenizers==0.20.3"
python -m pip install editdistance scikit-learn
```

`editdistance` and `scikit-learn` are imported by `inference.py` but are not listed in `requirements.txt`. The explicit Transformers pin targets the older `Seq2SeqTrainer` API used by the custom trainer; the open-ended lower bound in `requirements.txt` should not be interpreted as compatibility with every newer release. Install `deepspeed` separately if enabling `--use_deepspeed true`.

### 2. Prepare SLUE Data

Obtain the datasets and follow their access and preparation instructions from [SLUE on Hugging Face](https://huggingface.co/datasets/asapp/slue). The loaders expect **local TSV annotations and WAV audio**, not a Hugging Face `Dataset` object.

```text
data/
├── slue-voxpopuli/
│   ├── slue-voxpopuli_fine-tune.tsv
│   ├── slue-voxpopuli_dev.tsv
│   ├── fine-tune-wav/<id>.wav
│   └── dev-wav/<id>.wav
└── slue-voxceleb/
    ├── slue-voxceleb_fine-tune.tsv
    ├── slue-voxceleb_dev.tsv
    ├── fine-tune-wav/<id>.wav
    └── dev-wav/<id>.wav
```

| Dataset | Required TSV columns | Annotation format |
| --- | --- | --- |
| SLUE-VoxPopuli | `id`, `split`, `normalized_text`, `normalized_ner` | Python-literal list of `[entity_type, character_start, character_length]` entries |
| SLUE-VoxCeleb | `id`, `split`, `normalized_text`, `sentiment` | Sentiment label, e.g. `Positive`, `Negative`, or `Neutral` |

For example, `normalized_ner` can be `[['ORG', 5, 7]]` for `this council made great progress`. Use `[]` for an utterance without entities. Offsets must refer to `normalized_text`. The loader maps `GPE/LOC` to `PLACE`, numeric types to `QUANT`, and `DATE/TIME` to `WHEN`; it reverses the serialized entity list when constructing English training targets.

Each TSV must contain one split: audio is resolved as `<audio_root>/<split>-wav/<id>.wav`, using the first row's `split` value. Convert source audio to WAV if necessary. The loaders downmix audio to mono and resample to 16 kHz. SA examples labeled `Disagreement` or `<mixed>` are excluded.

### 3. Prepare the Task-Specific Processor

`train.py` loads the processor from the relative directory **`processor/whisper-small-Special`**. Despite that directory name, create it from the same Whisper family used for training. Run this from the repository root before starting a new English experiment:

```bash
python - <<'PY'
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-medium", language="en", task="transcribe"
)
tokens = ["[T/L]", "[NER]", "[SA]", "[S]", "[/S]"]
for label in ["PLACE", "QUANT", "ORG", "WHEN", "NORP", "PERSON", "LAW"]:
    tokens.extend([f"[{label}]", f"[/{label}]"])
processor.tokenizer.add_tokens(tokens)
assert all(len(processor.tokenizer.encode(t, add_special_tokens=False)) == 1 for t in tokens)
processor.save_pretrained("processor/whisper-small-Special")
PY
```

These markers are added as ordinary vocabulary tokens so that decoding does not discard them. Training resizes the model's token embeddings automatically. This creates a processor for **new training**; it does not reconstruct the unpublished original token-ID mapping. When loading an existing fine-tuned checkpoint, use the exact tokenizer used to train it.

## Training

The paper uses Whisper-medium, 100 epochs, a learning rate of `1e-5`, and batch size 32. The example below uses a per-device batch size of 4 and eight accumulation steps on one GPU for an effective training batch size of 32:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 train.py \
    --model openai/whisper-medium \
    --output_dir output/unified-whisper-medium \
    --language en \
    --epochs 100 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --ner_dataset slue \
    --sa_dataset slue \
    --ner_audio_root_path data/slue-voxpopuli \
    --sa_audio_root_path data/slue-voxceleb \
    --fixed_loss_weight false \
    --early_stop_epoch false \
    --frozen_encoder false \
    --gradient_checkpointing true \
    --fp16 true
```

Adjust batch sizes to available GPU memory. Effective batch size is `per_device_train_batch_size × gradient_accumulation_steps × number_of_GPUs`.

| Option | Behavior |
| --- | --- |
| `--fixed_loss_weight false` | Enable length-based dynamic weighting. |
| `--fixed_loss_weight true --fixed_asr_weight 0.5` | Use equal fixed ASR and task weights. |
| `--early_stop_epoch false` | Keep the selected weighting throughout training. When `true`, the code switches to `L_task + 0.05 * L_ASR` after epoch 20; it does **not** stop training. |
| `--frozen_encoder true` | Freeze the Whisper encoder. |
| `--ner_audio_root_path none` / `--sa_audio_root_path none` | Omit the corresponding task for a separate-model run. |
| `--cache_features true` | Cache extracted audio features on disk. |
| `--use_deepspeed true --deepspeed_stage 2` | Enable the optional DeepSpeed configuration. |

The bundled `train.sh` contains placeholder paths, uses fixed loss weights, and enables the post-epoch-20 loss switch. Edit it before use, or use the explicit command above. The command enables the paper's dynamic-weighting formulation; it is not an archived configuration for all reported runs. Validation loss controls checkpoint selection. Checkpoints and TensorBoard logs are saved under `--output_dir`.

## Inference and Evaluation

Select a saved `checkpoint-*` directory, which must contain the trained model and its tokenizer. `--processor_path` supplies the feature extractor; the tokenizer is always read from `--model_path`.

The following evaluates both tasks on the development sets:

```bash
python inference.py \
    --model_path output/unified-whisper-medium/checkpoint-STEP \
    --processor_path processor/whisper-small-Special \
    --language en \
    --ner_dataset slue \
    --ner_data_path data/slue-voxpopuli/slue-voxpopuli_dev.tsv \
    --ner_audio_root data/slue-voxpopuli \
    --sa_dataset slue \
    --sa_data_path data/slue-voxceleb/slue-voxceleb_dev.tsv \
    --sa_audio_root data/slue-voxceleb \
    --batch_size 4 \
    --beam_size 1 \
    --device cuda:0 \
    --output_dir inference_results
```

Replace `checkpoint-STEP` with an actual checkpoint. For test evaluation, provide the corresponding labeled test TSVs and audio directories; annotations are needed by this script to calculate metrics. Omit a task's data argument or set it to `none` to evaluate only the other task.

The script writes `ner_results.json` and `sa_results.json` beneath a checkpoint-specific subdirectory of `--output_dir`, including predictions, metrics, and performance summaries. It reports ASR WER, NER Micro-F1, and SA Macro-F1/accuracy. Compute the overall SLUE score from the percentage-valued metrics as:

```text
SLUE = ((100 - (WER_VP + WER_VC) / 2) + NER_Micro_F1 + SA_Macro_F1) / 3
```

### Implementation Notes

- The evaluation loop uses batched `model.generate()`. It does not force the requested `[NER]` or `[SA]` token after `[T/L]`. Although a custom decoder is included in `model.py`, the current evaluation loop does not call `inference_single_dynamic()`; `--use_dynamic_decoder` alone does not activate that decoding procedure.
- Runtime FLOPs are estimates, and the evaluation code assumes 10 seconds per utterance when calculating RTF. These diagnostics should not be treated as exact reproductions of the paper's efficiency measurements.
- Chinese data adapters are also included. With `--language zh`, the `aishell` NER selector reads `train.jsonl` / `valid.jsonl` with `sentence`, `path`, and `entity` fields; entities have the form `[start, end, text, type]` with `ORG`, `PER`, or `LOC`. The `ch-sims` SA selector reads `train.csv` / `val.csv` with `text`, `path`, and `annotation` columns. Chinese NER requires adding `[PER]`, `[/PER]`, `[LOC]`, and `[/LOC]` to the processor before training. The `aishell` selector is the code's adapter name, not a direct CNERTA downloader or preprocessing pipeline.

## Citation

If you use this work, please cite our NLPCC paper. The entry below uses the accepted title; proceedings-specific page numbers and DOI are omitted until available.

```bibtex
@inproceedings{sheng2026unifiedslu,
  title     = {Unified Spoken Language Understanding from Heterogeneous Data: Spoken NER and SA as a Case Study},
  author    = {Sheng, Zhichao and Zhou, Shilin and Gong, Chen and Li, Zhenghua},
  booktitle = {Natural Language Processing and Chinese Computing (NLPCC)},
  year      = {2026}
}
```

## Acknowledgments

This implementation builds on [Whisper](https://github.com/openai/whisper), [Hugging Face Transformers](https://github.com/huggingface/transformers), and the [SLUE benchmark](https://huggingface.co/datasets/asapp/slue). Please consult the original projects and datasets for their respective licenses and access conditions.

For questions, open an issue in this repository or contact **Zhichao Sheng** at [zcsheng@stu.suda.edu.cn](mailto:zcsheng@stu.suda.edu.cn).
