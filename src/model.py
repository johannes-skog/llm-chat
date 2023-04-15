from abc import abstractmethod
import os
from typing import List, Any,Dict
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
import peft
import lightning.pytorch as pl
import torchmetrics
import transformers
# import deepspeed
import numpy as np
from util import IGNORE_LOSS_ID
from checkpointing import CheckpointModel
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam


class Llama(torch.nn.Module):
    
    def __init__(self, from_pretrained: bool = False, **kwargs):
        
        super().__init__()
        
        if from_pretrained:

            self._model = transformers.LlamaForCausalLM.from_pretrained(
                "decapoda-research/llama-7b-hf",
                **kwargs
            )
            
        else:
            
            self._model = transformers.LlamaForCausalLM(
                config=transformers.LlamaConfig()
            )

        self._vobab_size = 32000

    def setup_lora(self):

        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )

        self._model = peft.get_peft_model(self._model, peft_config)
    
    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self._model(input_ids=input_ids, attention_mask=attention_mask).logits
            
class Gpt2(torch.nn.Module):
    
    def __init__(self, from_pretrained: bool = False, **kwargs):
        
        super().__init__()
        
        if from_pretrained:
            
            self._model = transformers.GPT2LMHeadModel.from_pretrained(
                "gpt2", is_decoder=True, **kwargs
            )
            
        else:
            
            self._model = transformers.GPT2LMHeadModel(
                config=transformers.GPT2Config()
            )

        self._vobab_size = 50257

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    def setup_lora(self):

        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            # target_modules=["c_attn", "c_proj"],
        )

        self._model = peft.get_peft_model(self._model, peft_config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self._model(input_ids=input_ids, attention_mask=attention_mask).logits
            
class GeneratorModel(pl.LightningModule):
    
    def __init__(
        self,
        T_mult: int = 2,
        T_0: int = 1000,
        learning_rate: float = 1e-5,
        lora: bool = False,
        inference: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 16,
        model: str = "decapoda-research/llama-7b-hf",
        lora_pretrained: str = None,
        lora_target_modules: List[str] = None,
        **kwargs,
    ):
        
        super().__init__()
        
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            model,
            **kwargs
        )

        self._learning_rate = learning_rate
        self._T_0 = T_0
        self._T_mult = T_mult
        self._inference = inference
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lora_target_modules = lora_target_modules
        self._lora_pretrained = lora_pretrained
        self._lora = lora

        self._loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=IGNORE_LOSS_ID,
            reduction="none"
        )

        if lora:
            self.setup_lora()
            if lora_pretrained:
                self.load_lora_pretrained(lora_pretrained)
     
    def setup_lora(self):

        self._model = peft.get_peft_model(
            self._model,
            peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM,
                inference_mode=self._inference,
                r=self._lora_r,
                lora_alpha=self._lora_alpha,
                lora_dropout=0.05,
                target_modules=self._lora_target_modules,
            )
        )

    def generate(self, *args, **kwargs):

        return self._model.generate(*args, **kwargs)
    
    def save_pretrained(self, *args, **kwargs):
        return self._model.save_pretrained(*args, **kwargs)

    def load_lora_pretrained(self, path: str):

        d = {k: v for k, v in torch.load(path)["state_dict"].items() if "lora" in k}
        self._model.load_state_dict(d, strict=False)

    def forward(self, *args, **kwargs):
        
        return self._model(*args, **kwargs, labels=None).logits

    def loss_function(self, logits: torch.Tensor, labels: torch.Tensor):

        vobab_size = logits.shape[-1]

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vobab_size)
        shift_labels = shift_labels.view(-1)

        l = self._loss_func(shift_logits, shift_labels)

        not_ignore_mask = (shift_labels != IGNORE_LOSS_ID)

        l = l.sum() / not_ignore_mask.float().sum()

        predicted_token = torch.argmax(shift_logits, dim=1)

        accuracy_all = (predicted_token == shift_labels).float().mean()

        accuracy = (predicted_token[not_ignore_mask] == shift_labels[not_ignore_mask]).float().mean()

        return l, accuracy, accuracy_all
    
    def training_step(self, batch, batch_idx):
        
        y = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        
        l, accuracy, accuracy_all = self.loss_function(
            logits=y, labels=batch["target_ids"]
        )
        
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        self.log("loss_train", l)
        self.log("accuracy_train", accuracy)
        self.log("accuracy_all_train", accuracy_all)
        
        return l

    def validation_step(self, batch, batch_idx):

        y = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        l, accuracy, accuracy_all = self.loss_function(
            logits=y, labels=batch["target_ids"]
        )
        
        self.log("loss_val", l)
        self.log("accuracy_val", accuracy)
        self.log("accuracy_all_val", accuracy_all)

        return l
        
    def configure_optimizers(self):

        self._opt = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._opt,
            T_0=self._T_0,
            T_mult=self._T_mult,
            eta_min=0,
            last_epoch=- 1,
            verbose=False
        )

        return [self._opt], [{"scheduler": lr_scheduler, "interval": "step"}]
