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

@dataclass
class TrainingStructure():

    TRAIN_DATASET: str = "train"
    VAL_DATASET: str = "val"

    optimizer: Any = None

    dataloaders: Dict = field(default_factory=dict)
    frequency: Dict = field(default_factory=dict)

    iters: Dict = field(default_factory=dict)

    writer: Any = None

    global_iteration: int = 0

@dataclass
class TrainingLoopStructure():

    current_iteration = None
    iterations: int = None
    save_freq: int = None

class NLPmodel(torch.nn.Module):

    def __init__(
        self,
        encoder: torch.nn.Module,
        tokenizer,
        model_folder: str = "models/ckp",
        device: str = None,
        lora: bool = False,
    ):

        super().__init__()

        self._encoder = encoder
        self._tokenizer = tokenizer
        self._model_folder = model_folder
        self._training_structure = TrainingStructure()

        self._device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device is None else device
    



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
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
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
            lora_alpha=27,
            lora_dropout=0.1,
        )

        self._model = peft.get_peft_model(self._model, peft_config)

        self._model.lm_head.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self._model(input_ids=input_ids, attention_mask=attention_mask).logits
            
class GeneratorModel(pl.LightningModule):
    
    def __init__(
        self,
        # model_name: str,
        base_model: torch.nn.Module,
        T_mult: int = 2,
        T_0: int = 1000,
        learning_rate: float = 1e-5,
        from_pretrained: bool = False,
        lora: bool = False,
        **kwargs,
    ):
        
        super().__init__()
        
        self._model = base_model

        self._vobab_size = self._model._vobab_size
        self._learning_rate = learning_rate
        self._T_0 = T_0
        self._T_mult = T_mult
        self._loss_func = torch.nn.CrossEntropyLoss()

        if lora:
            self.setup_lora()

    def forward(self, **kwargs):
        
        return self._model(**kwargs)

    def setup_lora(self):

        self._model.setup_lora()
    
    def loss(self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self._vobab_size)
        shift_labels = shift_labels.view(-1)
        shift_weights = shift_weights.view(-1)

        #print(shift_logits.shape, shift_labels.shape)
        
        l = self._loss_func(shift_logits, shift_labels)

        # l = (l * shift_weights).sum() / shift_weights.sum()

        return l
    
    def training_step(self, batch, batch_idx):
        
        y = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        
        l = self.loss(
            logits=y, labels=batch["target_ids"], weights=batch["target_weight"]
        )
        
        self.log("learning_rate", self._opt.param_groups[0]["lr"])
        self.log("loss_train", l)
        
        return l

    def validation_step(self, batch, batch_idx):

        y = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        l = self.loss(
            logits=y, labels=batch["target_ids"], weights=batch["target_weight"]
        )
        
        self.log("loss_val", l)

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
