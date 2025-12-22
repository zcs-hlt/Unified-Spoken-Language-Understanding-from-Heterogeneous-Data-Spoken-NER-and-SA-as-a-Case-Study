import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
from mask import mask_finished_preds, mask_finished_scores
from utils import get_logger
from typing import Optional, List

logger = get_logger(__name__)


class DynamicDecodeWhisper(WhisperForConditionalGeneration):
    def __init__(self, config, task_delimiter_token: Optional[str] = "[T/L]"):
        super().__init__(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.task_delimiter_token = task_delimiter_token
        self._task_delimiter_id: Optional[int] = None

    def set_task_delimiter_id(self, tokenizer):
        if self.task_delimiter_token:
            token_ids = tokenizer.convert_tokens_to_ids([self.task_delimiter_token])
            if token_ids and token_ids[0] != tokenizer.unk_token_id:
                self._task_delimiter_id = token_ids[0]
            else:
                logger.warning(f"Task delimiter token '{self.task_delimiter_token}' not found in tokenizer")
                self._task_delimiter_id = None

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past),)
        return reordered_past

    @torch.no_grad()
    def dynamic_decoder(self, feature, decoder_prompt_ids, beam_size, task_id, decode_max_len=448):
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, device=self.device)
        else:
            feature = feature.to(self.device)
            
        if not isinstance(decoder_prompt_ids, torch.Tensor):
            decoder_prompt_ids = torch.tensor(decoder_prompt_ids, device=self.device)
        else:
            decoder_prompt_ids = decoder_prompt_ids.to(self.device)
        
        encoder_out = self.model.encoder(feature.unsqueeze(0))[0]
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        batch_size = encoder_out.size(0)
        device = self.device
        running_size = batch_size * beam_size
        
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim
        )

        hyps = decoder_prompt_ids.unsqueeze(0).expand(running_size, -1).long()
        
        past_key_values = self.model.decoder(
            input_ids=hyps[:, :-1], 
            attention_mask=hyps[:, :-1].ne(self.model.config.eos_token_id), 
            encoder_hidden_states=encoder_out, 
            past_key_values=None, 
            use_cache=True
        )[1]
        
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1), dtype=torch.float, device=device)
        scores = scores.repeat([batch_size]).unsqueeze(1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool)

        for i in range(1, decode_max_len + 1):
            if end_flag.sum() == running_size:
                break
                
            s_y, past_key_values = self.model.decoder(
                input_ids=hyps[:, -1:],
                attention_mask=hyps.ne(self.model.config.eos_token_id),
                encoder_hidden_states=encoder_out,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=False
            )
            
            logp = self.proj_out(s_y[:, -1]).log_softmax(-1)
            
            top_k_logp, top_k_index = logp.topk(beam_size)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.model.config.eos_token_id)
            
            scores = scores + top_k_logp
            scores = scores.view(batch_size, beam_size * beam_size)
            scores, offset_k_index = scores.topk(k=beam_size)
            scores = scores.view(-1, 1)
            
            base_k_index = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size])
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)
            
            best_k_pred = torch.index_select(top_k_index.view(-1), dim=-1, index=best_k_index)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
            past_key_values = self._reorder_cache(past_key_values, best_hyps_index)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)

            if self._task_delimiter_id is not None and int(hyps[0][-1]) == self._task_delimiter_id:
                hyps = torch.cat((hyps, torch.full((hyps.size(0), 1), task_id, device=device)), dim=1)
                past_key_values = self._reorder_cache(past_key_values, best_hyps_index)

            end_flag = torch.eq(hyps[:, -1], self.model.config.eos_token_id).view(-1, 1)
        
        scores = scores.view(batch_size, beam_size)
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        
        return best_hyps


def create_model(config, processor, use_dynamic_decoder: bool = False):
    if use_dynamic_decoder:
        model = DynamicDecodeWhisper.from_pretrained(config.model)
        model.set_task_delimiter_id(processor.tokenizer)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(config.model)
    
    model.resize_token_embeddings(len(processor.tokenizer))
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    
    if config.frozen_encoder:
        model.freeze_encoder()
        logger.info("Encoder frozen for training")
    
    return model 