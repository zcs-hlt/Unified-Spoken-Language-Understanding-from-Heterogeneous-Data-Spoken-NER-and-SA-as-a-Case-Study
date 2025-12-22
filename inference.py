"""
UniSLU Inference and Evaluation Script

Features:
- Supports NER and SA (Sentiment Analysis) tasks
- Batch inference for maximum speed
- Custom beam search decoder support
- Saves inference results to files
- Computes WER, F1 metrics
- Performance metrics: throughput, latency, RTF, memory usage
"""

import os
import re
import json
import argparse
import time
import torch
import torchaudio
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import editdistance
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from model import DynamicDecodeWhisper
from utils import get_logger, setup_logging

# Try to import English text normalizer
try:
    from whisper.normalizers.english import EnglishTextNormalizer
    NORMALIZER = EnglishTextNormalizer()
except ImportError:
    logger.warning("EnglishTextNormalizer not available, using simple lowercase normalization")
    NORMALIZER = lambda x: x.lower().strip()

logger = get_logger(__name__)


# ============================================================
# Performance Metrics
# ============================================================

@dataclass
class PerformanceMetrics:
    """Container for inference performance metrics"""
    total_samples: int = 0
    total_inference_time: float = 0.0  # seconds
    total_audio_duration: float = 0.0  # seconds
    peak_gpu_memory_mb: float = 0.0
    model_parameters: int = 0
    model_size_mb: float = 0.0
    batch_times: List[float] = field(default_factory=list)
    
    @property
    def throughput(self) -> float:
        """Samples per second"""
        if self.total_inference_time > 0:
            return self.total_samples / self.total_inference_time
        return 0.0
    
    @property
    def latency_per_sample_ms(self) -> float:
        """Average latency per sample in milliseconds"""
        if self.total_samples > 0:
            return (self.total_inference_time / self.total_samples) * 1000
        return 0.0
    
    @property
    def real_time_factor(self) -> float:
        """Real-time factor (RTF): inference_time / audio_duration
        RTF < 1 means faster than real-time"""
        if self.total_audio_duration > 0:
            return self.total_inference_time / self.total_audio_duration
        return 0.0
    
    def to_dict(self) -> Dict:
        return {
            "total_samples": self.total_samples,
            "total_inference_time_sec": round(self.total_inference_time, 3),
            "total_audio_duration_sec": round(self.total_audio_duration, 3),
            "throughput_samples_per_sec": round(self.throughput, 2),
            "latency_per_sample_ms": round(self.latency_per_sample_ms, 2),
            "real_time_factor": round(self.real_time_factor, 4),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 2),
            "model_parameters": self.model_parameters,
            "model_size_mb": round(self.model_size_mb, 2),
        }


def count_parameters(model: torch.nn.Module) -> Tuple[int, float]:
    """Count model parameters and estimate size in MB"""
    total_params = sum(p.numel() for p in model.parameters())
    # Assuming fp16 (2 bytes per parameter)
    size_mb = total_params * 2 / (1024 * 1024)
    return total_params, size_mb


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def estimate_flops_whisper(model_config, seq_len: int = 448, audio_frames: int = 3000) -> int:
    """
    Estimate FLOPs for Whisper model inference.
    This is a rough estimation based on transformer architecture.
    
    FLOPs for transformer:
    - Self-attention: 4 * seq_len^2 * d_model
    - FFN: 8 * seq_len * d_model * d_ff
    - Per layer, per encoder/decoder
    """
    d_model = model_config.d_model
    encoder_layers = model_config.encoder_layers
    decoder_layers = model_config.decoder_layers
    d_ff = d_model * 4  # Typical FFN expansion
    
    # Encoder FLOPs (process audio_frames)
    encoder_attention = 4 * audio_frames * audio_frames * d_model * encoder_layers
    encoder_ffn = 8 * audio_frames * d_model * d_ff * encoder_layers
    encoder_flops = encoder_attention + encoder_ffn
    
    # Decoder FLOPs (generate seq_len tokens, averaged)
    avg_seq = seq_len // 2  # Average sequence length during generation
    decoder_self_attention = 4 * avg_seq * avg_seq * d_model * decoder_layers
    decoder_cross_attention = 4 * avg_seq * audio_frames * d_model * decoder_layers
    decoder_ffn = 8 * avg_seq * d_model * d_ff * decoder_layers
    decoder_flops = decoder_self_attention + decoder_cross_attention + decoder_ffn
    
    return encoder_flops + decoder_flops


def format_flops(flops: int) -> str:
    """Format FLOPs to human readable string"""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops} FLOPs"


# ============================================================
# Metrics
# ============================================================

def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Safe division avoiding inf/nan"""
    numerator = np.array(numerator)
    denominator = np.array(denominator)
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1
    return numerator / denominator


def compute_wer(refs: List[str], hyps: List[str]) -> float:
    """Compute Word Error Rate (for English)"""
    n_words, n_errors = 0, 0
    for ref, hyp in zip(refs, hyps):
        ref_words, hyp_words = ref.split(), hyp.split()
        n_words += len(ref_words)
        n_errors += editdistance.eval(ref_words, hyp_words)
    return float(safe_divide(n_errors, n_words)) * 100


def compute_cer(refs: List[str], hyps: List[str]) -> float:
    """Compute Character Error Rate (for Chinese)"""
    n_chars, n_errors = 0, 0
    for ref, hyp in zip(refs, hyps):
        # Remove spaces for Chinese character-level comparison
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        n_chars += len(ref_chars)
        n_errors += editdistance.eval(ref_chars, hyp_chars)
    return float(safe_divide(n_errors, n_chars)) * 100


def normalize_chinese_text(text: str) -> str:
    """Normalize Chinese text for CER evaluation
    
    For Chinese CER calculation:
    - Remove Whisper special tokens
    - Remove punctuation (both Chinese and English)
    - Remove extra whitespace
    """
    # Remove Whisper special tokens that might remain
    text = re.sub(r'<\|[^|]+\|>', '', text)
    
    # Remove Chinese punctuation
    chinese_punc = '，。！？、；：""''【】《》（）—…～·'
    for p in chinese_punc:
        text = text.replace(p, '')
    
    # Remove English punctuation
    english_punc = ',.!?;:\'"[]{}()<>-_=+/\\|@#$%^&*~`'
    for p in english_punc:
        text = text.replace(p, '')
    
    # Remove all whitespace
    text = ''.join(text.split())
    
    return text


def make_distinct(label_lst: List[Tuple]) -> List[Tuple]:
    """Make entity labels distinct by adding count"""
    tag2cnt, new_tag_lst = {}, []
    for tag_item in label_lst:
        _ = tag2cnt.setdefault(tag_item, 0)
        tag2cnt[tag_item] += 1
        tag, wrd = tag_item
        new_tag_lst.append((tag, wrd, tag2cnt[tag_item]))
    return new_tag_lst


def get_ner_scores(all_gt: List, all_predictions: List) -> Dict:
    """Compute NER scores (precision, recall, F1)"""
    stats = {}
    
    for gt, pred in zip(all_gt, all_predictions):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        
        for item in gt:
            if len(item) >= 3:
                type_name, entity_info1, entity_info2 = item[0], item[1], item[2]
                entities_true[type_name].add((entity_info1, entity_info2))
        
        for item in pred:
            if len(item) >= 3:
                type_name, entity_info1, entity_info2 = item[0], item[1], item[2]
                entities_pred[type_name].add((entity_info1, entity_info2))
        
        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
        
        for tag_name in target_names:
            stats.setdefault(tag_name, {"tp": [], "gt_cnt": [], "pred_cnt": []})
            entities_true_type = entities_true.get(tag_name, set())
            entities_pred_type = entities_pred.get(tag_name, set())
            stats[tag_name]["tp"].append(len(entities_true_type & entities_pred_type))
            stats[tag_name]["pred_cnt"].append(len(entities_pred_type))
            stats[tag_name]["gt_cnt"].append(len(entities_true_type))
    
    # Compute metrics
    metrics = {}
    num_correct, num_gt, num_pred = 0, 0, 0
    
    for tag_name, tag_stats in stats.items():
        tp = np.sum(tag_stats["tp"])
        gt_cnt = np.sum(tag_stats["gt_cnt"])
        pred_cnt = np.sum(tag_stats["pred_cnt"])
        
        precision = float(safe_divide(tp, pred_cnt))
        recall = float(safe_divide(tp, gt_cnt))
        fscore = float(safe_divide(2 * precision * recall, precision + recall)) if (precision + recall) > 0 else 0.0
        
        metrics[tag_name] = {"precision": precision, "recall": recall, "fscore": fscore}
        num_correct += tp
        num_pred += pred_cnt
        num_gt += gt_cnt
    
    # Overall micro
    precision = float(safe_divide(num_correct, num_pred))
    recall = float(safe_divide(num_correct, num_gt))
    fscore = float(safe_divide(2 * precision * recall, precision + recall)) if (precision + recall) > 0 else 0.0
    metrics["overall_micro"] = {"precision": precision, "recall": recall, "fscore": fscore}
    
    return metrics


def compute_sa_accuracy(refs: List[str], hyps: List[str]) -> float:
    """Compute Sentiment Analysis accuracy"""
    if len(refs) == 0:
        return 0.0
    correct = sum(1 for r, h in zip(refs, hyps) if r.lower() == h.lower())
    return correct / len(refs) * 100


def compute_sa_metrics(refs: List[str], hyps: List[str]) -> Dict:
    """Compute comprehensive Sentiment Analysis metrics including F1, Precision, Recall
    
    Args:
        refs: List of reference sentiment labels
        hyps: List of predicted sentiment labels
    
    Returns:
        Dictionary containing accuracy, macro/micro F1, precision, recall, and per-class metrics
    """
    if len(refs) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "micro_f1": 0.0,
            "per_class": {}
        }
    
    # Normalize labels
    refs_normalized = [r.lower() for r in refs]
    hyps_normalized = [h.lower() for h in hyps]
    
    # Valid sentiment labels
    valid_labels = ['positive', 'negative', 'neutral']
    
    # Predictions should already be valid labels (handled in evaluate_sa with penalty)
    hyps_mapped = []
    for h in hyps_normalized:
        if h in valid_labels:
            hyps_mapped.append(h)
        else:
            # Fallback: if still unknown, map to neutral (shouldn't happen with penalty logic)
            hyps_mapped.append('neutral')
    
    # Compute accuracy
    correct = sum(1 for r, h in zip(refs_normalized, hyps_mapped) if r == h)
    accuracy = correct / len(refs_normalized) * 100
    
    # Get classification report
    try:
        report = classification_report(
            refs_normalized, 
            hyps_mapped, 
            labels=valid_labels,
            target_names=['Positive', 'Negative', 'Neutral'],
            output_dict=True,
            zero_division=0
        )
        
        macro_f1 = report['macro avg']['f1-score'] * 100
        macro_precision = report['macro avg']['precision'] * 100
        macro_recall = report['macro avg']['recall'] * 100
        micro_f1 = report['weighted avg']['f1-score'] * 100  # weighted avg approximates micro
        
        per_class = {
            'Positive': {
                'precision': report['Positive']['precision'] * 100,
                'recall': report['Positive']['recall'] * 100,
                'f1': report['Positive']['f1-score'] * 100,
                'support': report['Positive']['support']
            },
            'Negative': {
                'precision': report['Negative']['precision'] * 100,
                'recall': report['Negative']['recall'] * 100,
                'f1': report['Negative']['f1-score'] * 100,
                'support': report['Negative']['support']
            },
            'Neutral': {
                'precision': report['Neutral']['precision'] * 100,
                'recall': report['Neutral']['recall'] * 100,
                'f1': report['Neutral']['f1-score'] * 100,
                'support': report['Neutral']['support']
            }
        }
    except Exception as e:
        logger.warning(f"Error computing classification report: {e}")
        macro_f1 = macro_precision = macro_recall = micro_f1 = 0.0
        per_class = {}
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
        "per_class": per_class
    }


# ============================================================
# Text Processing
# ============================================================

def split_asr_and_task(text: str) -> Tuple[str, str]:
    """Split text into ASR part and task-specific part"""
    delimiter = "[T/L]"
    if delimiter in text:
        parts = text.split(delimiter, 1)
        return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
    return text.strip(), ""


def extract_ner_entities(tagged_text: str, normalize: bool = True, language: str = "en") -> List[Tuple[str, str]]:
    """Extract NER entities from tagged text
    
    Args:
        tagged_text: Text with NER tags like [PLACE]entity[/PLACE]
        normalize: Whether to normalize entity text
        language: Language for normalization ('en' uses EnglishTextNormalizer, 'zh' uses strip only)
    
    Returns:
        List of (tag, entity_text) tuples
    """
    # Valid NER tags (exclude SA tags like S)
    # English: PLACE, QUANT, WHEN, ORG, NORP, PERSON, LAW
    # Chinese: ORG, PER, LOC
    valid_ner_tags = {'PLACE', 'QUANT', 'WHEN', 'ORG', 'NORP', 'PERSON', 'LAW', 'PER', 'LOC'}
    
    pattern = r'\[([A-Z]+)\](.*?)\[/\1\]'
    matches = re.findall(pattern, tagged_text)
    
    entities = []
    for tag, phrase in matches:
        if tag in valid_ner_tags:
            # Normalize entity text based on language
            if normalize:
                if language == "zh":
                    phrase = phrase.strip()  # Chinese: just strip whitespace
                else:
                    phrase = NORMALIZER(phrase)  # English: use EnglishTextNormalizer
            entities.append((tag, phrase))
    
    return entities


def extract_sentiment(tagged_text: str) -> Optional[str]:
    """Extract sentiment label from tagged text"""
    pattern = r'\[S\](\w+)\[/S\]'
    match = re.search(pattern, tagged_text)
    return match.group(1) if match else None


# ============================================================
# Dataset
# ============================================================

class InferenceDataset(Dataset):
    """Dataset for inference supporting both NER and SA tasks"""
    
    def __init__(self, file_path: str, root_path: str, processor: WhisperProcessor, task: str = "ner"):
        self.processor = processor
        self.task = task.lower()
        self._resampler_cache = {}
        
        # Load data
        self.data = pd.read_csv(file_path, sep='\t')
        self.split = self.data['split'][0] + '-wav'
        self.root = os.path.join(root_path, self.split)
        
        # Process based on task
        if self.task == "ner":
            self._load_ner_data()
        elif self.task == "sa":
            self._load_sa_data()
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _load_ner_data(self):
        """Load NER dataset"""
        self.audio_names = self.data['id'].values.tolist()
        self.texts = self.data['normalized_text'].values.tolist()
        self.ner_labels = self.data['normalized_ner'].values.tolist()
        self.references = self._build_ner_references()
    
    def _load_sa_data(self):
        """Load SA dataset"""
        audio_names = self.data['id'].values.tolist()
        texts = self.data['normalized_text'].values.tolist()
        sentiments = self.data['sentiment'].values.tolist()
        
        # Filter out invalid sentiments
        self.audio_names = []
        self.texts = []
        self.sentiments = []
        
        for name, text, sentiment in zip(audio_names, texts, sentiments):
            if sentiment not in ['Disagreement', '<mixed>']:
                self.audio_names.append(name)
                self.texts.append(text)
                self.sentiments.append(sentiment)
        
        self.references = self.sentiments
    
    def _build_ner_references(self) -> List[List[Tuple]]:
        """Build NER reference entities with text normalization"""
        import ast
        place_tags = ['GPE', 'LOC']
        quant_tags = ['CARDINAL', 'MONEY', 'ORDINAL', 'PERCENT', 'QUANTITY']
        when_tags = ['DATE', 'TIME']
        other_tags = ['ORG', 'NORP', 'PERSON', 'LAW']
        
        references = []
        for idx, ner_label in enumerate(self.ner_labels):
            entities = []
            if isinstance(ner_label, str):
                try:
                    for item in ast.literal_eval(ner_label):
                        tag, start, length = item[0], item[1], item[2]
                        text = self.texts[idx][start:start + length]
                        
                        # Map to simplified tags
                        if tag in place_tags:
                            mapped_tag = "PLACE"
                        elif tag in quant_tags:
                            mapped_tag = "QUANT"
                        elif tag in when_tags:
                            mapped_tag = "WHEN"
                        elif tag in other_tags:
                            mapped_tag = tag
                        else:
                            continue
                        
                        # Normalize entity text for consistent matching
                        normalized_text = NORMALIZER(text)
                        entities.append((mapped_tag, normalized_text))
                except:
                    pass
            references.append(make_distinct(entities))
        return references
    
    def _get_resampler(self, orig_sr: int, target_sr: int = 16000):
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resampler_cache[orig_sr]
    
    def __len__(self):
        return len(self.audio_names)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.root, f"{self.audio_names[idx]}.wav")
        audio, sample_rate = torchaudio.load(audio_path)
        
        # Handle multi-channel
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != 16000:
            audio = self._get_resampler(sample_rate)(audio)
        
        audio = audio.squeeze(0)
        feature = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        
        return {
            "audio_id": self.audio_names[idx],
            "input_features": feature,
            "reference_text": self.texts[idx],
            "reference": self.references[idx] if hasattr(self, 'references') else None,
        }


@dataclass
class InferenceCollator:
    """Collator for batched inference"""
    processor: Any
    
    def __call__(self, features: List[Dict]) -> Dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        batch["audio_ids"] = [f["audio_id"] for f in features]
        batch["reference_texts"] = [f["reference_text"] for f in features]
        batch["references"] = [f["reference"] for f in features]
        
        return batch


class ChineseNERInferenceDataset(Dataset):
    """Chinese NER dataset for inference (AISHELL-NER format)"""
    
    def __init__(self, file_path: str, processor: WhisperProcessor):
        self.processor = processor
        self._resampler_cache = {}
        
        # Load JSONL data
        self.samples = []
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples from Chinese NER test dataset")
    
    def _get_resampler(self, orig_sr: int, target_sr: int = 16000):
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resampler_cache[orig_sr]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample['path']
        sentence = sample['sentence']
        entities = sample.get('entity', [])
        
        # Build reference entities (sorted by position, NER types: ORG, PER, LOC)
        # For Chinese, just strip whitespace (matching extract_ner_entities behavior)
        ref_entities = []
        sorted_entities = sorted(entities, key=lambda x: x[0]) if entities else []
        for entity in sorted_entities:
            entity_text = entity[2].strip()  # Strip whitespace for consistent matching
            entity_type = entity[3]
            if entity_type in ['ORG', 'PER', 'LOC']:
                ref_entities.append((entity_type, entity_text))
        ref_entities = make_distinct(ref_entities)
        
        # Load audio
        audio, sample_rate = torchaudio.load(audio_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = self._get_resampler(sample_rate)
            audio = resampler(audio)
        audio = audio.squeeze(0)
        
        feature = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        
        return {
            "audio_id": sample.get('audio', str(idx)),
            "input_features": feature,
            "reference_text": sentence,
            "reference": ref_entities,
        }


class ChineseSAInferenceDataset(Dataset):
    """Chinese SA dataset for inference (CH-SIMS format)"""
    
    VALID_SENTIMENTS = {'Positive', 'Negative', 'Neutral'}
    
    def __init__(self, file_path: str, processor: WhisperProcessor):
        self.processor = processor
        self._resampler_cache = {}
        
        # Load CSV data
        df = pd.read_csv(file_path)
        self.samples = []
        
        for _, row in df.iterrows():
            sentiment = row['annotation']
            if sentiment in self.VALID_SENTIMENTS:
                self.samples.append({
                    'audio_id': f"{row['video_id']}_{row['clip_id']}",
                    'text': row['text'],
                    'sentiment': sentiment,
                    'path': row['path']
                })
        
        logger.info(f"Loaded {len(self.samples)} samples from Chinese SA test dataset")
    
    def _get_resampler(self, orig_sr: int, target_sr: int = 16000):
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resampler_cache[orig_sr]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample['path']
        
        # Load audio
        audio, sample_rate = torchaudio.load(audio_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = self._get_resampler(sample_rate)
            audio = resampler(audio)
        audio = audio.squeeze(0)
        
        feature = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        
        return {
            "audio_id": sample['audio_id'],
            "input_features": feature,
            "reference_text": sample['text'],
            "reference": sample['sentiment'],
        }


# ============================================================
# Inference Engine
# ============================================================

class InferenceEngine:
    """High-performance inference engine for UniSLU with performance tracking"""
    
    def __init__(
        self,
        model_path: str,
        processor_path: str,
        device: str = "cuda:0",
        use_dynamic_decoder: bool = False,
        beam_size: int = 5,
        language: str = "en",
    ):
        self.device = torch.device(device)
        self.use_dynamic_decoder = use_dynamic_decoder
        self.beam_size = beam_size
        self.language = language
        
        # Map language code
        lang_map = {"en": "english", "zh": "chinese"}
        whisper_language = lang_map.get(language, language)
        
        # Load processor (feature_extractor from base model, tokenizer from checkpoint)
        logger.info(f"Loading processor from {processor_path}")
        logger.info(f"Loading tokenizer from {model_path} (language={whisper_language})")
        
        feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_path)
        tokenizer = WhisperTokenizer.from_pretrained(model_path, language=whisper_language, task="transcribe")
        self.processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        
        # Load model with optimizations
        logger.info(f"Loading model from {model_path}")
        if use_dynamic_decoder:
            self.model = DynamicDecodeWhisper.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Use fp16 for faster inference
            )
            self.model.set_task_delimiter_id(self.processor.tokenizer)
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Use fp16 for faster inference
            )
        
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.model.to(self.device)
        self.model.eval()
        
        # Disable gradient computation globally for inference
        torch.set_grad_enabled(False)
        
        # Get task token IDs
        self.ner_token_id = self.processor.tokenizer.convert_tokens_to_ids(["[NER]"])[0]
        self.sa_token_id = self.processor.tokenizer.convert_tokens_to_ids(["[SA]"])[0]
        
        # Model statistics
        self.model_params, self.model_size_mb = count_parameters(self.model)
        self.estimated_flops = estimate_flops_whisper(self.model.config)
        
        logger.info("Inference engine initialized")
        logger.info(f"  Model parameters: {self.model_params:,}")
        logger.info(f"  Model size: {self.model_size_mb:.2f} MB")
        logger.info(f"  Estimated FLOPs per sample: {format_flops(self.estimated_flops)}")
    
    @torch.no_grad()
    def inference_batch(self, batch: Dict, task: str = "ner") -> Tuple[List[str], float]:
        """Batch inference using generate()
        
        Returns:
            Tuple of (transcriptions, inference_time_seconds)
        """
        input_features = batch["input_features"].to(self.device, dtype=torch.float16)
        
        # Synchronize GPU before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # Generate with correct language setting
        generated_ids = self.model.generate(
            input_features,
            max_new_tokens=256,
            num_beams=self.beam_size,
            language=self.language,  # Use configured language (en/zh)
            task="transcribe",
            use_cache=True,
            return_timestamps=False,
        )
        
        # Synchronize GPU after generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = time.perf_counter() - start_time
        
        # Decode - keep special tokens to see NER/SA tags
        transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        
        # Clean up transcriptions (remove start tokens but keep task tokens)
        cleaned = []
        for t in transcriptions:
            t = t.replace("<|startoftranscript|>", "")
            t = t.replace("<|en|>", "")
            t = t.replace("<|zh|>", "")  # Remove Chinese language token
            t = t.replace("<|transcribe|>", "")
            t = t.replace("<|notimestamps|>", "")
            t = t.replace("<|endoftext|>", "")
            cleaned.append(t.strip())
        
        return cleaned, inference_time
    
    @torch.no_grad()
    def inference_single_dynamic(self, feature: np.ndarray, task: str = "ner") -> str:
        """Single sample inference using custom beam search decoder"""
        task_id = self.ner_token_id if task == "ner" else self.sa_token_id
        
        hyps = self.model.dynamic_decoder(
            feature,
            self.decoder_prompt_ids,
            self.beam_size,
            task_id,
        )
        
        transcription = self.processor.decode(hyps[0], skip_special_tokens=True)
        return transcription


# ============================================================
# Evaluation
# ============================================================

def evaluate_ner(
    engine: InferenceEngine,
    dataset: InferenceDataset,
    batch_size: int = 16,
    output_file: Optional[str] = None,
    language: str = "en",
) -> Tuple[Dict, PerformanceMetrics]:
    """Evaluate NER task with performance metrics"""
    logger.info(f"Evaluating NER on {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=InferenceCollator(engine.processor),
        pin_memory=True,
    )
    
    # Initialize performance tracking
    perf_metrics = PerformanceMetrics()
    perf_metrics.model_parameters = engine.model_params
    perf_metrics.model_size_mb = engine.model_size_mb
    reset_gpu_memory_stats()
    
    results = []
    all_ref_entities = []
    all_pred_entities = []
    all_ref_texts = []
    all_pred_texts = []
    
    # Warmup run (first batch may be slower due to CUDA initialization)
    warmup_done = False
    
    for batch in tqdm(dataloader, desc="NER Inference"):
        transcriptions, batch_time = engine.inference_batch(batch, task="ner")
        
        # Skip first batch for timing (warmup)
        if warmup_done:
            perf_metrics.batch_times.append(batch_time)
            perf_metrics.total_inference_time += batch_time
            perf_metrics.total_samples += len(transcriptions)
        else:
            warmup_done = True
            # Still count samples but note it's warmup
            perf_metrics.total_samples += len(transcriptions)
            perf_metrics.total_inference_time += batch_time
        
        # Estimate audio duration (30 seconds per sample is Whisper's max)
        # Actual duration would need to be computed from audio files
        perf_metrics.total_audio_duration += len(transcriptions) * 10  # Assume 10s avg
        
        for i, transcription in enumerate(transcriptions):
            audio_id = batch["audio_ids"][i]
            ref_text = batch["reference_texts"][i]
            ref_entities = batch["references"][i]
            
            # Parse prediction
            # Note: No need to reverse - data preprocessing already converts to correct order
            pred_asr, pred_task = split_asr_and_task(transcription)
            pred_entities = extract_ner_entities(pred_task, normalize=True, language=language)
            pred_entities_distinct = make_distinct(pred_entities)
            
            results.append({
                "audio_id": audio_id,
                "reference_text": ref_text,
                "predicted_text": pred_asr,
                "reference_entities": [(e[0], e[1]) for e in ref_entities] if ref_entities else [],
                "predicted_entities": pred_entities,
                "full_prediction": transcription,
            })
            
            # Normalize text for error rate computation
            if language == "zh":
                all_ref_texts.append(normalize_chinese_text(ref_text))
                all_pred_texts.append(normalize_chinese_text(pred_asr))
            else:
                all_ref_texts.append(NORMALIZER(ref_text))
                all_pred_texts.append(NORMALIZER(pred_asr))
            all_ref_entities.append(ref_entities if ref_entities else [])
            all_pred_entities.append(pred_entities_distinct)
    
    # Record peak GPU memory
    perf_metrics.peak_gpu_memory_mb = get_gpu_memory_mb()
    
    # Compute metrics - use CER for Chinese, WER for English
    if language == "zh":
        error_rate = compute_cer(all_ref_texts, all_pred_texts)
        error_rate_name = "cer"
    else:
        error_rate = compute_wer(all_ref_texts, all_pred_texts)
        error_rate_name = "wer"
    
    ner_scores = get_ner_scores(all_ref_entities, all_pred_entities)
    
    metrics = {
        error_rate_name: error_rate,
        "ner_micro_f1": ner_scores["overall_micro"]["fscore"] * 100,
        "ner_micro_precision": ner_scores["overall_micro"]["precision"] * 100,
        "ner_micro_recall": ner_scores["overall_micro"]["recall"] * 100,
        "per_tag_scores": {k: v for k, v in ner_scores.items() if k != "overall_micro"},
    }
    
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_data = {
            "metrics": metrics,
            "performance": perf_metrics.to_dict(),
            "estimated_flops_per_sample": format_flops(engine.estimated_flops),
            "results": results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    
    return metrics, perf_metrics


def evaluate_sa(
    engine: InferenceEngine,
    dataset: InferenceDataset,
    batch_size: int = 16,
    output_file: Optional[str] = None,
    language: str = "en",
) -> Tuple[Dict, PerformanceMetrics]:
    """Evaluate Sentiment Analysis task with performance metrics"""
    logger.info(f"Evaluating SA on {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=InferenceCollator(engine.processor),
        pin_memory=True,
    )
    
    # Initialize performance tracking
    perf_metrics = PerformanceMetrics()
    perf_metrics.model_parameters = engine.model_params
    perf_metrics.model_size_mb = engine.model_size_mb
    reset_gpu_memory_stats()
    
    results = []
    all_ref_texts = []
    all_pred_texts = []
    all_ref_sentiments = []
    all_pred_sentiments = []
    
    # Warmup run
    warmup_done = False
    
    for batch in tqdm(dataloader, desc="SA Inference"):
        transcriptions, batch_time = engine.inference_batch(batch, task="sa")
        
        # Skip first batch for timing (warmup)
        if warmup_done:
            perf_metrics.batch_times.append(batch_time)
            perf_metrics.total_inference_time += batch_time
            perf_metrics.total_samples += len(transcriptions)
        else:
            warmup_done = True
            perf_metrics.total_samples += len(transcriptions)
            perf_metrics.total_inference_time += batch_time
        
        # Estimate audio duration
        perf_metrics.total_audio_duration += len(transcriptions) * 10  # Assume 10s avg
        
        for i, transcription in enumerate(transcriptions):
            audio_id = batch["audio_ids"][i]
            ref_text = batch["reference_texts"][i]
            ref_sentiment = batch["references"][i]
            
            # Parse prediction
            pred_asr, pred_task = split_asr_and_task(transcription)
            pred_sentiment_raw = extract_sentiment(pred_task)
            
            # Handle empty predictions: assign wrong label as penalty (matching reference script)
            if pred_sentiment_raw is None:
                # Penalty: assign a deliberately wrong label
                if ref_sentiment == "Neutral":
                    pred_sentiment = "Positive"
                elif ref_sentiment == "Negative":
                    pred_sentiment = "Neutral"
                elif ref_sentiment == "Positive":
                    pred_sentiment = "Negative"
                else:
                    pred_sentiment = "Neutral"
            else:
                pred_sentiment = pred_sentiment_raw
            
            results.append({
                "audio_id": audio_id,
                "reference_text": ref_text,
                "predicted_text": pred_asr,
                "reference_sentiment": ref_sentiment,
                "predicted_sentiment": pred_sentiment,
                "full_prediction": transcription,
            })
            
            # Normalize text for error rate computation
            if language == "zh":
                all_ref_texts.append(normalize_chinese_text(ref_text))
                all_pred_texts.append(normalize_chinese_text(pred_asr))
            else:
                all_ref_texts.append(NORMALIZER(ref_text))
                all_pred_texts.append(NORMALIZER(pred_asr))
            all_ref_sentiments.append(ref_sentiment if ref_sentiment else "Unknown")
            all_pred_sentiments.append(pred_sentiment)
    
    # Record peak GPU memory
    perf_metrics.peak_gpu_memory_mb = get_gpu_memory_mb()
    
    # Compute metrics - use CER for Chinese, WER for English
    if language == "zh":
        error_rate = compute_cer(all_ref_texts, all_pred_texts)
        error_rate_name = "cer"
    else:
        error_rate = compute_wer(all_ref_texts, all_pred_texts)
        error_rate_name = "wer"
    
    sa_metrics = compute_sa_metrics(all_ref_sentiments, all_pred_sentiments)
    
    metrics = {
        error_rate_name: error_rate,
        "sa_accuracy": sa_metrics["accuracy"],
        "sa_macro_f1": sa_metrics["macro_f1"],
        "sa_macro_precision": sa_metrics["macro_precision"],
        "sa_macro_recall": sa_metrics["macro_recall"],
        "sa_micro_f1": sa_metrics["micro_f1"],
        "sa_per_class": sa_metrics["per_class"],
    }
    
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_data = {
            "metrics": metrics,
            "performance": perf_metrics.to_dict(),
            "estimated_flops_per_sample": format_flops(engine.estimated_flops),
            "results": results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    
    return metrics, perf_metrics


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="UniSLU Inference and Evaluation")
    
    # Model paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to processor (default: same as model)")
    
    # Language and dataset type
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"],
                       help="Language: 'en' (English, use WER) or 'zh' (Chinese, use CER)")
    parser.add_argument("--ner_dataset", type=str, default="slue",
                       help="NER dataset type: 'slue' (English) or 'aishell' (Chinese)")
    parser.add_argument("--sa_dataset", type=str, default="slue",
                       help="SA dataset type: 'slue' (English) or 'ch-sims' (Chinese)")
    
    # Data paths (use 'none' to disable a task)
    parser.add_argument("--ner_data_path", type=str, default=None, 
                       help="Path to NER test file (TSV for slue, JSONL for aishell, 'none' to disable)")
    parser.add_argument("--ner_audio_root", type=str, default=None, help="Root path for NER audio files")
    parser.add_argument("--sa_data_path", type=str, default=None, 
                       help="Path to SA test file (TSV for slue, CSV for ch-sims, 'none' to disable)")
    parser.add_argument("--sa_audio_root", type=str, default=None, help="Root path for SA audio files")
    
    # Inference settings
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--use_dynamic_decoder", action="store_true", help="Use custom beam search decoder")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")
    
    return parser.parse_args()


def log_performance_metrics(perf: PerformanceMetrics, task_name: str):
    """Log performance metrics"""
    logger.info(f"{task_name} Performance Metrics:")
    logger.info(f"  Total samples: {perf.total_samples}")
    logger.info(f"  Total inference time: {perf.total_inference_time:.2f}s")
    logger.info(f"  Throughput: {perf.throughput:.2f} samples/sec")
    logger.info(f"  Latency per sample: {perf.latency_per_sample_ms:.2f}ms")
    logger.info(f"  Real-time factor (RTF): {perf.real_time_factor:.4f}")
    logger.info(f"  Peak GPU memory: {perf.peak_gpu_memory_mb:.2f} MB")
    logger.info(f"  Model parameters: {perf.model_parameters:,}")
    logger.info(f"  Model size: {perf.model_size_mb:.2f} MB")


def build_output_path(output_dir: str, model_path: str) -> str:
    """Build output path based on model checkpoint path
    
    Args:
        output_dir: Base output directory (e.g., ./inference_results)
        model_path: Path to model checkpoint (e.g., model-output/test-new-code/checkpoint-2100)
    
    Returns:
        Full output path (e.g., ./inference_results/test-new-code/checkpoint-2100-res)
    
    Example:
        model_path = "model-output/test-new-code/checkpoint-2100"
        output_dir = "./inference_results"
        result = "./inference_results/test-new-code/checkpoint-2100-res"
    """
    # Normalize path (remove trailing slashes)
    model_path = model_path.rstrip('/')
    
    # Split path into parts
    path_parts = model_path.split('/')
    
    # Find the index after "model-output" or similar base directory
    # We want to extract: sub-directory/checkpoint-xxx
    # Common patterns: model-output/xxx/checkpoint-xxx, /path/to/model-output/xxx/checkpoint-xxx
    
    # Try to find "model-output" or similar marker
    base_markers = ['model-output', 'models', 'checkpoints', 'output']
    start_idx = 0
    
    for i, part in enumerate(path_parts):
        if part.lower() in base_markers:
            start_idx = i + 1
            break
    
    # If no marker found, use last 2 parts (sub-dir/checkpoint)
    if start_idx == 0 and len(path_parts) >= 2:
        start_idx = len(path_parts) - 2
    elif start_idx == 0:
        start_idx = len(path_parts) - 1
    
    # Extract sub-path
    sub_parts = path_parts[start_idx:]
    
    # Build result path: sub-dir/checkpoint-xxx-res
    if sub_parts:
        # Last part is the checkpoint name, add "-res" suffix
        sub_parts[-1] = sub_parts[-1] + "-res"
        result_subpath = '/'.join(sub_parts)
    else:
        result_subpath = "results"
    
    return os.path.join(output_dir, result_subpath)


def main():
    args = parse_args()
    
    setup_logging()
    
    # Use model path as processor path if not specified
    processor_path = args.processor_path or args.model_path
    
    # Initialize engine
    engine = InferenceEngine(
        model_path=args.model_path,
        processor_path=processor_path,
        device=args.device,
        use_dynamic_decoder=args.use_dynamic_decoder,
        beam_size=args.beam_size,
        language=args.language,
    )
    
    # Build output directory based on checkpoint path
    # e.g., model-output/test-new-code/checkpoint-2100 -> ./inference_results/test-new-code/checkpoint-2100-res
    result_dir = build_output_path(args.output_dir, args.model_path)
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {result_dir}")
    
    # Evaluate NER
    ner_path = args.ner_data_path
    if ner_path and ner_path.lower() != 'none':
        logger.info("=" * 60)
        logger.info(f"Evaluating NER Task ({args.ner_dataset})")
        logger.info("=" * 60)
        
        # Load dataset based on type
        if args.ner_dataset == "aishell":
            ner_dataset = ChineseNERInferenceDataset(
                args.ner_data_path,
                engine.processor,
            )
        else:  # slue
            ner_dataset = InferenceDataset(
                args.ner_data_path,
                args.ner_audio_root,
                engine.processor,
                task="ner",
            )
        
        ner_metrics, ner_perf = evaluate_ner(
            engine,
            ner_dataset,
            batch_size=args.batch_size,
            output_file=os.path.join(result_dir, "ner_results.json"),
            language=args.language,
        )
        
        # Log results (use CER for Chinese, WER for English)
        error_rate_name = "CER" if args.language == "zh" else "WER"
        error_rate_key = "cer" if args.language == "zh" else "wer"
        logger.info(f"NER Results:")
        logger.info(f"  {error_rate_name}: {ner_metrics[error_rate_key]:.2f}%")
        logger.info(f"  Micro F1: {ner_metrics['ner_micro_f1']:.2f}%")
        logger.info(f"  Micro Precision: {ner_metrics['ner_micro_precision']:.2f}%")
        logger.info(f"  Micro Recall: {ner_metrics['ner_micro_recall']:.2f}%")
        logger.info("-" * 40)
        log_performance_metrics(ner_perf, "NER")
    
    # Evaluate SA
    sa_path = args.sa_data_path
    if sa_path and sa_path.lower() != 'none':
        logger.info("=" * 60)
        logger.info(f"Evaluating SA Task ({args.sa_dataset})")
        logger.info("=" * 60)
        
        # Load dataset based on type
        if args.sa_dataset == "ch-sims":
            sa_dataset = ChineseSAInferenceDataset(
                args.sa_data_path,
                engine.processor,
            )
        else:  # slue
            sa_dataset = InferenceDataset(
                args.sa_data_path,
                args.sa_audio_root,
                engine.processor,
                task="sa",
            )
        
        sa_metrics, sa_perf = evaluate_sa(
            engine,
            sa_dataset,
            batch_size=args.batch_size,
            output_file=os.path.join(result_dir, "sa_results.json"),
            language=args.language,
        )
        
        # Log results (use CER for Chinese, WER for English)
        error_rate_name = "CER" if args.language == "zh" else "WER"
        error_rate_key = "cer" if args.language == "zh" else "wer"
        logger.info(f"SA Results:")
        logger.info(f"  {error_rate_name}: {sa_metrics[error_rate_key]:.2f}%")
        logger.info(f"  Accuracy: {sa_metrics['sa_accuracy']:.2f}%")
        logger.info(f"  Macro F1: {sa_metrics['sa_macro_f1']:.2f}%")
        logger.info(f"  Macro Precision: {sa_metrics['sa_macro_precision']:.2f}%")
        logger.info(f"  Macro Recall: {sa_metrics['sa_macro_recall']:.2f}%")
        if sa_metrics.get('sa_per_class'):
            logger.info(f"  Per-class metrics:")
            for cls_name, cls_metrics in sa_metrics['sa_per_class'].items():
                logger.info(f"    {cls_name}: F1={cls_metrics['f1']:.2f}%, P={cls_metrics['precision']:.2f}%, R={cls_metrics['recall']:.2f}%, N={cls_metrics['support']}")
        logger.info("-" * 40)
        log_performance_metrics(sa_perf, "SA")
    
    logger.info("=" * 60)
    logger.info("Evaluation completed!")
    logger.info(f"Estimated FLOPs per sample: {format_flops(engine.estimated_flops)}")


if __name__ == "__main__":
    main()

