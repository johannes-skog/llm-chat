from abc import abstractmethod
from typing import List
import torch
import peft
import lightning.pytorch as pl
import transformers
import numpy as np
from util import IGNORE_LOSS_ID
from checkpointing import CheckpointModel
import torch
import numpy as np
from typing import Union, List
from transformers import AutoTokenizer

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def _generate_next_token(
    model,
    tokenizer,
    input_ids,
    temperature=1.0,
    top_p=0.95,
    device='cpu'
):
    
    # Ensure the model is in evaluation mode
    model.eval()

    # Move input tensor to the specified device
    input_ids = torch.Tensor([input_ids]).to(device).long()
    
    # Generate output with the model
    with torch.no_grad():
        logits = model(input_ids)

    # Apply temperature scaling to logits
    logits = logits[:, -1, :] / temperature
    
    probs = torch.softmax(logits, dim=-1)
  
    next_token_ids = sample_top_p(probs, top_p).item()
    
    if next_token_ids == tokenizer.eos_token_id:
        return None

    return next_token_ids

def generate_next_tokens(
    model: torch.nn.Module,
    tokenizer,
    promt: str,
    temperature: float =1.0,
    top_p: float =0.95,
    max_length: int = 100,
    device: str = 'cpu',
):
        
    input_ids = tokenizer(promt)["input_ids"]
    
    s = len(input_ids)

    try: 
        from IPython.display import clear_output
    except ImportError:
        clear_output = None

    for i in range(max_length):

        next_token = _generate_next_token(
            input_ids=input_ids,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            top_p=top_p,
            device=device
        )

        if next_token is None:
            break

        input_ids.append(next_token)

        if clear_output is not None: clear_output(wait=True)
        
        decoded = tokenizer.decode(input_ids[s:])

        print(decoded, end = "\r")
        
    return decoded


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

        w1 = {name: param[0][0].item() for name, param in self.named_parameters() if "lora" in name}

        d = {k: v for k, v in torch.load(path).items() if "lora" in k}
        self._model.load_state_dict(d, strict=False)

        w2 = {name: param[0][0].item() for name, param in self.named_parameters() if "lora" in name}

        equals = [w1[k] == w2[k]  for k in w2.keys()]

        assert any(equals) is False, "Lora weights are not loaded"

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
