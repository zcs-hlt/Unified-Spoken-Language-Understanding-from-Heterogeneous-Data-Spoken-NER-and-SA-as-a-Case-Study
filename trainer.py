import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from utils import get_logger

logger = get_logger(__name__)


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, processor=None, early_stop_epoch=True,
                 fixed_loss_weight=False, fixed_asr_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.early_stop_epoch = early_stop_epoch
        self.fixed_loss_weight = fixed_loss_weight
        self.fixed_asr_weight = fixed_asr_weight
        self.fixed_task_weight = 1.0 - fixed_asr_weight
        self.task_tokens = ["[NER]", "[SA]"]
        self.target_token_ids = None

    def _init_target_token_ids(self, device):
        if self.target_token_ids is None:
            self.target_token_ids = torch.tensor(
                self.processor.tokenizer.convert_tokens_to_ids(self.task_tokens), 
                device=device
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        self._init_target_token_ids(model.device)
        labels = inputs.pop('labels')
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        asr_logits, task_logits, asr_labels, task_labels = self.split_logits(logits, labels)
        
        loss_asr = F.cross_entropy(
            asr_logits.view(-1, asr_logits.size(-1)), 
            asr_labels.view(-1), 
            ignore_index=-100
        )
        
        if (task_labels == -100).all():
            loss_task = torch.zeros(1, device=task_logits.device, requires_grad=False).squeeze()
        else:
            loss_task = F.cross_entropy(
                task_logits.view(-1, task_logits.size(-1)), 
                task_labels.view(-1), 
                ignore_index=-100
            )
        
        loss = self._compute_dynamic_loss(asr_logits, task_logits, asr_labels, task_labels, loss_asr, loss_task)
        return (loss, outputs) if return_outputs else loss

    def _compute_dynamic_loss(self, asr_logits, task_logits, asr_labels, task_labels, loss_asr, loss_task):
        if self.early_stop_epoch and self.state.epoch > 20:
            return loss_task + 0.05 * loss_asr
        
        if loss_task.item() == 0.0:
            return loss_asr
        
        if self.fixed_loss_weight:
            return self.fixed_asr_weight * loss_asr + self.fixed_task_weight * loss_task
        
        batch_size = asr_labels.size(0)
        device = loss_asr.device
        
        asr_counts = (asr_labels != -100).sum(dim=1).float()
        task_counts = (task_labels != -100).sum(dim=1).float()
        sample_totals = asr_counts + task_counts
        
        valid_mask = sample_totals > 0
        if not valid_mask.any():
            return loss_asr
        
        asr_weights = torch.where(valid_mask, task_counts / sample_totals, torch.zeros_like(asr_counts))
        task_weights = torch.where(valid_mask, asr_counts / sample_totals, torch.zeros_like(task_counts))
        
        asr_loss_per_token = F.cross_entropy(
            asr_logits.view(-1, asr_logits.size(-1)),
            asr_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        ).view(batch_size, -1)
        
        task_loss_per_token = F.cross_entropy(
            task_logits.view(-1, task_logits.size(-1)),
            task_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        ).view(batch_size, -1)
        
        asr_valid_counts = (asr_labels != -100).sum(dim=1).float().clamp(min=1)
        task_valid_counts = (task_labels != -100).sum(dim=1).float().clamp(min=1)
        
        sample_asr_losses = asr_loss_per_token.sum(dim=1) / asr_valid_counts
        sample_task_losses = task_loss_per_token.sum(dim=1) / task_valid_counts
        
        task_all_ignored = (task_labels == -100).all(dim=1)
        sample_task_losses = torch.where(task_all_ignored, torch.zeros_like(sample_task_losses), sample_task_losses)
        
        sample_losses = task_weights * sample_task_losses + asr_weights * sample_asr_losses
        weighted_losses = sample_losses * sample_totals
        
        total_tokens = sample_totals[valid_mask].sum()
        total_loss = weighted_losses[valid_mask].sum()
        
        return total_loss / total_tokens if total_tokens > 0 else loss_asr

    def split_logits(self, logits, labels):
        if self.target_token_ids.device != labels.device:
            self.target_token_ids = self.target_token_ids.to(labels.device)
            
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        is_task_token = torch.isin(labels, self.target_token_ids)
        has_task_token = is_task_token.any(dim=1)
        
        first_task_indices = torch.where(
            is_task_token,
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1),
            torch.full((batch_size, seq_len), seq_len, device=device)
        ).min(dim=1).values
        
        first_task_indices = torch.where(has_task_token, first_task_indices, 
                                         torch.full_like(first_task_indices, seq_len))
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        asr_mask = positions <= first_task_indices.unsqueeze(1)
        task_mask = positions > first_task_indices.unsqueeze(1)
        
        asr_lengths = (first_task_indices + 1).clamp(max=seq_len)
        task_lengths = (seq_len - first_task_indices - 1).clamp(min=1)
        
        max_asr_len = asr_lengths.max().item()
        max_task_len = task_lengths.max().item()
        
        asr_logits = torch.full((batch_size, max_asr_len, vocab_size), -100.0, device=device, dtype=logits.dtype)
        task_logits = torch.full((batch_size, max_task_len, vocab_size), -100.0, device=device, dtype=logits.dtype)
        asr_labels = torch.full((batch_size, max_asr_len), -100, device=device, dtype=labels.dtype)
        task_labels = torch.full((batch_size, max_task_len), -100, device=device, dtype=labels.dtype)
        
        for batch_idx in range(batch_size):
            asr_len = asr_lengths[batch_idx].item()
            task_len = task_lengths[batch_idx].item()
            task_start = first_task_indices[batch_idx].item() + 1
            
            asr_logits[batch_idx, :asr_len] = logits[batch_idx, :asr_len]
            asr_labels[batch_idx, :asr_len] = labels[batch_idx, :asr_len]
            
            if has_task_token[batch_idx] and task_start < seq_len:
                actual_task_len = min(task_len, seq_len - task_start)
                task_logits[batch_idx, :actual_task_len] = logits[batch_idx, task_start:task_start + actual_task_len]
                task_labels[batch_idx, :actual_task_len] = labels[batch_idx, task_start:task_start + actual_task_len]

        return asr_logits, task_logits, asr_labels, task_labels

    def log(self, logs):
        if hasattr(self.optimizer, 'param_groups'):
            logs['learning_rate'] = self.optimizer.param_groups[0]['lr']
        logs['epoch'] = self.state.epoch
        logs['step'] = self.state.global_step
        super().log(logs) 