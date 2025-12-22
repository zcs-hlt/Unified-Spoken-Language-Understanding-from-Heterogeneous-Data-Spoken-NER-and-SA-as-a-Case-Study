import ast
import os
import re
import hashlib
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import WhisperProcessor
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from utils import get_logger

logger = get_logger(__name__)


class FeatureCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Feature cache enabled at: {cache_dir}")
    
    def _get_cache_path(self, audio_path: str) -> str:
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{path_hash}.npy")
    
    def get(self, audio_path: str) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        
        cache_path = self._get_cache_path(audio_path)
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                return None
        return None
    
    def put(self, audio_path: str, feature: np.ndarray):
        if not self.enabled:
            return
        
        cache_path = self._get_cache_path(audio_path)
        try:
            np.save(cache_path, feature)
        except Exception as e:
            logger.warning(f"Failed to cache feature: {e}")


class NERData(Dataset):
    def __init__(self, file_path: str, root_path: str, processor: WhisperProcessor, 
                 asr_only: bool, split_token: bool, cache_features: bool = False,
                 cache_dir: Optional[str] = None):
        self.place = ['GPE', 'LOC']
        self.quant = ['CARDINAL', 'MONEY', 'ORDINAL', 'PERCENT', 'QUANTITY']
        self.when = ['DATE', 'TIME']
        self.others = ['ORG', 'NORP', 'PERSON', 'LAW']
        
        self.char_label_dict = {
            "PLACE": "_[", "QUANT": "_(", "ORG": "_{", "WHEN": "_$", 
            "NORP": "_&", "PERSON": "_%", "LAW": "_#"
        }
        
        self.split_token = split_token
        self.processor = processor
        self.asr_only = asr_only
        self._resampler_cache = {}
        
        if cache_dir is None:
            cache_dir = os.path.join(root_path, '.feature_cache')
        self.feature_cache = FeatureCache(cache_dir, enabled=cache_features)
        
        self.data = pd.read_csv(file_path, sep='\t')
        self.split = self.data['split'][0] + '-wav'
        self.audio_name = self.data['id'].values.copy()
        self.root = root_path + '/' + self.split + '/'
        
        labels = self.data['normalized_ner'].values.copy()
        self.labels = self.data['normalized_text'].values.copy()
        
        if not asr_only:
            self._process_ner_labels(labels)
    
    def _process_ner_labels(self, labels):
        for idx, label in enumerate(labels):
            if not isinstance(label, str):
                continue
                
            tmp_label = ''
            label_list = ast.literal_eval(label)
            
            if self.split_token:
                tmp_label = self._process_split_token_labels(label_list, idx)
            else:
                self._process_char_labels(label_list, idx)
            
            self.labels[idx] = self.labels[idx] + '[T/L][NER]' + self._reverse_sequence(tmp_label)
    
    def _process_split_token_labels(self, label_list, idx):
        tmp_label = ''
        for l in label_list:
            entity_text = self.labels[idx][l[1]:l[1] + l[2]]
            
            if l[0] in self.place:
                tmp_label += f'[PLACE]{entity_text}[/PLACE]'
            elif l[0] in self.quant:
                tmp_label += f'[QUANT]{entity_text}[/QUANT]'
            elif l[0] in self.when:
                tmp_label += f'[WHEN]{entity_text}[/WHEN]'
            elif l[0] in self.others:
                tmp_label += f'[{l[0]}]{entity_text}[/{l[0]}]'
        
        return tmp_label
    
    def _process_char_labels(self, label_list, idx):
        for l in label_list:
            entity_text = self.labels[idx][l[1]:l[1] + l[2]]
            
            if l[0] in self.place:
                char_label = self.char_label_dict["PLACE"]
            elif l[0] in self.quant:
                char_label = self.char_label_dict["QUANT"]
            elif l[0] in self.when:
                char_label = self.char_label_dict["WHEN"]
            elif l[0] in self.others:
                char_label = self.char_label_dict[l[0]]
            else:
                continue
            
            self.labels[idx] = (self.labels[idx][:l[1]] + char_label + 
                              entity_text + ']_' + self.labels[idx][l[1] + l[2]:])
    
    def _reverse_sequence(self, text):
        if not text:
            return text
        
        pattern = r'\[([A-Z]+)\](.*?)\[/\1\]'
        matches = re.findall(pattern, text)
        
        if not matches:
            return text
        
        reversed_matches = matches[::-1]
        return "".join([f"[{tag}]{phrase}[/{tag}]" for tag, phrase in reversed_matches])
    
    def __len__(self):
        return len(self.data)
    
    def _get_resampler(self, orig_sr: int, target_sr: int = 16000):
        """Get or create a cached resampler"""
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resampler_cache[orig_sr]
    
    def __getitem__(self, idx):
        path = self.audio_name[idx] + '.wav'
        full_path = self.root + path
        text = self.labels[idx]
        label = self.processor.tokenizer(text).input_ids
        
        # Try to get cached feature first
        cached_feature = self.feature_cache.get(full_path)
        if cached_feature is not None:
            return {'text': text, 'input_features': cached_feature, 'labels': label}
        
        # Load and preprocess audio
        audio, sample_rate = torchaudio.load(full_path)
        
        # Handle multi-channel audio: average channels
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if needed (use cached resampler)
        if sample_rate != 16000:
            resampler = self._get_resampler(sample_rate)
            audio = resampler(audio)
        
        # Squeeze to 1D for feature extractor
        audio = audio.squeeze(0)
        
        feature = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        
        # Cache the feature for future use
        self.feature_cache.put(full_path, feature)
        
        return {'text': text, 'input_features': feature, 'labels': label}


class SAData(Dataset):
    def __init__(self, file_path: str, root_path: str, processor: WhisperProcessor, 
                 asr_only: bool, split_token: bool, cache_features: bool = False,
                 cache_dir: Optional[str] = None):
        self.processor = processor
        self.split_token = split_token
        self.asr_only = asr_only
        self._resampler_cache = {}
        
        if cache_dir is None:
            cache_dir = os.path.join(root_path, '.feature_cache')
        self.feature_cache = FeatureCache(cache_dir, enabled=cache_features)
        
        self.data = pd.read_csv(file_path, sep='\t')
        self.split = self.data['split'][0] + '-wav'
        audio_name_all = self.data['id'].values.copy()
        self.audio_name = []
        self.root = root_path + '/' + self.split + '/'
        
        labels = self.data['sentiment'].values.copy()
        labels_all = self.data['normalized_text'].values.copy()
        self.labels = []
        
        for idx, label in enumerate(labels):
            if label not in ['Disagreement', '<mixed>']:
                self.audio_name.append(audio_name_all[idx])
                
                if asr_only:
                    self.labels.append(labels_all[idx])
                else:
                    sentiment_label = f"[S]{label}[/S]"
                    if split_token:
                        self.labels.append(labels_all[idx] + "[T/L][SA]" + sentiment_label)
                    else:
                        self.labels.append("[SA]" + labels_all[idx] + sentiment_label)
    
    def __len__(self):
        return len(self.labels)
    
    def _get_resampler(self, orig_sr: int, target_sr: int = 16000):
        """Get or create a cached resampler"""
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resampler_cache[orig_sr]
    
    def __getitem__(self, idx):
        path = self.audio_name[idx] + '.wav'
        full_path = self.root + path
        text = self.labels[idx]
        label = self.processor.tokenizer(text).input_ids
        
        # Try to get cached feature first
        cached_feature = self.feature_cache.get(full_path)
        if cached_feature is not None:
            return {'text': text, 'input_features': cached_feature, 'labels': label}
        
        # Load and preprocess audio
        audio, sample_rate = torchaudio.load(full_path)
        
        # Handle multi-channel audio: average channels
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if needed (use cached resampler)
        if sample_rate != 16000:
            resampler = self._get_resampler(sample_rate)
            audio = resampler(audio)
        
        # Squeeze to 1D for feature extractor
        audio = audio.squeeze(0)
        
        feature = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        
        # Cache the feature for future use
        self.feature_cache.put(full_path, feature)
        
        return {'text': text, 'input_features': feature, 'labels': label}


class ChineseNERData(Dataset):
    def __init__(self, file_path: str, processor: WhisperProcessor,
                 asr_only: bool = False, cache_features: bool = False,
                 cache_dir: Optional[str] = None):
        self.processor = processor
        self.asr_only = asr_only
        self._resampler_cache = {}
        
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(file_path), '.feature_cache_zh_ner')
        self.feature_cache = FeatureCache(cache_dir, enabled=cache_features)
        
        self.samples = []
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples from Chinese NER dataset")
    
    def _format_ner_label(self, sentence: str, entities: list) -> str:
        if not entities:
            return sentence + "[T/L][NER]"
        
        sorted_entities = sorted(entities, key=lambda x: x[0])
        ner_tags = ""
        for entity in sorted_entities:
            entity_text = entity[2]
            entity_type = entity[3]
            
            if entity_type in ['ORG', 'PER', 'LOC']:
                ner_tags += f"[{entity_type}]{entity_text}[/{entity_type}]"
        
        return sentence + "[T/L][NER]" + ner_tags
    
    def __len__(self):
        return len(self.samples)
    
    def _get_resampler(self, orig_sr: int, target_sr: int = 16000):
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resampler_cache[orig_sr]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sentence = sample['sentence']
        entities = sample.get('entity', [])
        audio_path = sample['path']
        
        # Format text
        if self.asr_only:
            text = sentence
        else:
            text = self._format_ner_label(sentence, entities)
        
        label = self.processor.tokenizer(text).input_ids
        
        # Try cached feature
        cached_feature = self.feature_cache.get(audio_path)
        if cached_feature is not None:
            return {'text': text, 'input_features': cached_feature, 'labels': label}
        
        # Load audio
        audio, sample_rate = torchaudio.load(audio_path)
        
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = self._get_resampler(sample_rate)
            audio = resampler(audio)
        
        audio = audio.squeeze(0)
        
        feature = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        
        self.feature_cache.put(audio_path, feature)
        
        return {'text': text, 'input_features': feature, 'labels': label}


class ChineseSAData(Dataset):
    VALID_SENTIMENTS = {'Positive', 'Negative', 'Neutral'}
    
    def __init__(self, file_path: str, processor: WhisperProcessor,
                 asr_only: bool = False, cache_features: bool = False,
                 cache_dir: Optional[str] = None):
        self.processor = processor
        self.asr_only = asr_only
        self._resampler_cache = {}
        
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(file_path), '.feature_cache_zh_sa')
        self.feature_cache = FeatureCache(cache_dir, enabled=cache_features)
        
        self.samples = []
        df = pd.read_csv(file_path)
        
        skipped = 0
        for _, row in df.iterrows():
            sentiment = row['annotation']
            
            # Skip invalid sentiments
            if sentiment not in self.VALID_SENTIMENTS:
                skipped += 1
                continue
            
            self.samples.append({
                'text': row['text'],
                'sentiment': sentiment,
                'path': row['path']
            })
        
        logger.info(f"Loaded {len(self.samples)} samples from Chinese SA dataset (skipped {skipped})")
    
    def _format_sa_label(self, text: str, sentiment: str) -> str:
        return f"{text}[T/L][SA][S]{sentiment}[/S]"
    
    def __len__(self):
        return len(self.samples)
    
    def _get_resampler(self, orig_sr: int, target_sr: int = 16000):
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resampler_cache[orig_sr]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Format text
        if self.asr_only:
            text = sample['text']
        else:
            text = self._format_sa_label(sample['text'], sample['sentiment'])
        
        label = self.processor.tokenizer(text).input_ids
        audio_path = sample['path']
        
        # Try cached feature
        cached_feature = self.feature_cache.get(audio_path)
        if cached_feature is not None:
            return {'text': text, 'input_features': cached_feature, 'labels': label}
        
        # Load audio
        audio, sample_rate = torchaudio.load(audio_path)
        
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = self._get_resampler(sample_rate)
            audio = resampler(audio)
        
        audio = audio.squeeze(0)
        
        feature = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        
        self.feature_cache.put(audio_path, feature)
        
        return {'text': text, 'input_features': feature, 'labels': label}


class CombinedDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = [d for d in datasets if d is not None]
        self.lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = []
        
        total = 0
        for length in self.lengths:
            total += length
            self.cumulative_lengths.append(total)
        
        self.total_length = total
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of range")
        
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                if i == 0:
                    return self.datasets[i][idx]
                else:
                    return self.datasets[i][idx - self.cumulative_lengths[i-1]]
        
        raise IndexError("Index out of range")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch


def load_datasets(config, processor):
    cache_features = getattr(config, 'cache_features', False)
    ner_dataset_type = getattr(config, 'ner_dataset', 'slue')
    sa_dataset_type = getattr(config, 'sa_dataset', 'slue')
    
    train_datasets = []
    dev_datasets = []
    
    ner_path = getattr(config, 'ner_audio_root_path', '')
    if ner_path and ner_path.lower() != 'none':
        if ner_dataset_type == "slue":
            logger.info("Loading NER dataset: SLUE-VoxPopuli (English)")
            ner_train = NERData(
                f'{ner_path}/slue-voxpopuli_fine-tune.tsv',
                ner_path, processor, config.asr_only, config.split_token,
                cache_features=cache_features
            )
            ner_dev = NERData(
                f'{ner_path}/slue-voxpopuli_dev.tsv',
                ner_path, processor, config.asr_only, config.split_token,
                cache_features=cache_features
            )
        elif ner_dataset_type == "aishell":
            logger.info("Loading NER dataset: AISHELL-NER (Chinese)")
            ner_train = ChineseNERData(
                f'{ner_path}/train.jsonl',
                processor, config.asr_only, cache_features=cache_features
            )
            ner_dev = ChineseNERData(
                f'{ner_path}/valid.jsonl',
                processor, config.asr_only, cache_features=cache_features
            )
        else:
            raise ValueError(f"Unknown NER dataset type: {ner_dataset_type}")
        
        train_datasets.append(ner_train)
        dev_datasets.append(ner_dev)
        logger.info(f"  NER - Train: {len(ner_train)}, Dev: {len(ner_dev)}")
    
    sa_path = getattr(config, 'sa_audio_root_path', '')
    if sa_path and sa_path.lower() != 'none':
        if sa_dataset_type == "slue":
            logger.info("Loading SA dataset: SLUE-VoxCeleb (English)")
            sa_train = SAData(
                f'{sa_path}/slue-voxceleb_fine-tune.tsv',
                sa_path, processor, config.asr_only, config.split_token,
                cache_features=cache_features
            )
            sa_dev = SAData(
                f'{sa_path}/slue-voxceleb_dev.tsv',
                sa_path, processor, config.asr_only, config.split_token,
                cache_features=cache_features
            )
        elif sa_dataset_type == "ch-sims":
            logger.info("Loading SA dataset: CH-SIMS v2 (Chinese)")
            sa_train = ChineseSAData(
                f'{sa_path}/train.csv',
                processor, config.asr_only, cache_features=cache_features
            )
            sa_dev = ChineseSAData(
                f'{sa_path}/val.csv',
                processor, config.asr_only, cache_features=cache_features
            )
        else:
            raise ValueError(f"Unknown SA dataset type: {sa_dataset_type}")
        
        train_datasets.append(sa_train)
        dev_datasets.append(sa_dev)
        logger.info(f"  SA - Train: {len(sa_train)}, Dev: {len(sa_dev)}")
    
    train_dataset = CombinedDataset(*train_datasets)
    dev_dataset = CombinedDataset(*dev_datasets)
    
    logger.info(f"Total - Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")
    
    return train_dataset, dev_dataset 