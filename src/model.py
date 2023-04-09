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
from util import IGNORE_LOSS_ID

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
        base_model: torch.nn.Module,
        T_mult: int = 2,
        T_0: int = 1000,
        learning_rate: float = 1e-5,
        lora: bool = False,
        **kwargs,
    ):
        
        super().__init__()
        
        self._model = base_model

        self._vobab_size = self._model._vobab_size
        self._learning_rate = learning_rate
        self._T_0 = T_0
        self._T_mult = T_mult
        self._loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=IGNORE_LOSS_ID,
            reduction="none"
        )

        if lora:
            self._model.setup_lora()

    def forward(self, **kwargs):
        
        return self._model(**kwargs)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor):

        # print(logits.shape, labels.shape, labels.dtype)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self._vobab_size)
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
        
        l, accuracy, accuracy_all = self.loss(
            logits=y, labels=batch["target_ids"]
        )
        
        self.log("learning_rate", self._opt.param_groups[0]["lr"])
        self.log("loss_train", l)
        self.log("accuracy_train", accuracy)
        self.log("accuracy_all_train", accuracy_all)
        
        return l

    def validation_step(self, batch, batch_idx):

        y = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        l, accuracy, accuracy_all = self.loss(
            logits=y, labels=batch["target_ids"]
        )
        
        self.log("loss_val", l)
        self.log("accuracy_val", accuracy)
        self.log("accuracy_all_val", accuracy_all)

        return l
        
    def configure_optimizers(self):

        # import deepspeed
        #self._opt = deepspeed.ops.adam.DeepSpeedCPUAdam(self.parameters(), lr=self._learning_rate)

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
