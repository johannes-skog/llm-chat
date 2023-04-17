from typing import Any, Dict, List
import torch
import torch.nn as nn
import logging
import torch.utils.checkpoint as checkpoint_utils
import deepspeed


class Conv1dWCkp(nn.Conv1d):
    def __init__(self, **kwargs):
        super(Conv1dWCkp, self).__init__(**kwargs)

    def forward(self, x):
        return deepspeed.checkpointing.checkpoint(super().forward, x)

class Conv2dWCkp(nn.Conv2d):
    def __init__(self, **kwargs):
        super(Conv2dWCkp, self).__init__(**kwargs)

    def forward(self, x):
        return deepspeed.checkpointing.checkpoint(super().forward, x)

class LinearWCkp(nn.Linear):
    def __init__(self, **kwargs):
        super(LinearWCkp, self).__init__(**kwargs)

    def forward(self, x):
        return deepspeed.checkpointing.checkpoint(super().forward, x)


class CheckpointModel(torch.nn.Module):

    def __init__(self, model: nn.Module, input_shape = None):
        super().__init__()
        self._model = model
        self._input_shape = input_shape

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)
    
    def replace_conv(
        self,
        include_names: List[str] = None,
    ):
        self._replace_conv(
            self._model,
            include_names=include_names,
        )

    def _replace_conv(
        self,
        module: torch.nn.Module,
        include_names: List[str] = None,
    ):
        for name, child in module.named_children():
            new_child = None

            if isinstance(child, nn.Conv1d):
                
                include = []
                
                if include_names is not None:
                    if name in include_names:
                        logging.info(f"Layer {name} is in the list of permitted names.")
                        include.append(True)

                if not any(include):
                    continue
        
                new_child = Conv1dWCkp(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                )

            elif isinstance(child, nn.Conv2d):
                
                include = []

                if include_names is not None:
                    if name in include_names:
                        logging.info(f"Layer {name} is in the list of permitted names.")
                        include.append(True)

                if not any(include):
                    continue

                new_child = Conv2dWCkp(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                )
            

            if new_child is not None:
                new_child.to(child.weight.device, child.weight.dtype)
                new_child.load_state_dict(child.state_dict())
                setattr(module, name, new_child)

            self._replace_conv(
                module=child,
                include_names=include_names,
            )
            
    def replace_linear(
        self,
        include_names: List[str] = None,
        output_limit: int = None,
        flops_limit: int = None,
    ):
        self._replace_linear(
            self._model,
            include_names=include_names,
            output_limit=output_limit,
            flops_limit=flops_limit,
        )

    def _get_flops_linear(self, module: torch.nn.Module, d: List[Any] = None):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                out_features, in_features = child.weight.size()
                flops = int(in_features * out_features)
                d.append(flops)
                logging.info(f"Layer nn.Linear {name} has {flops} flops.")
            self._get_flops_linear(child, d)
        
    def get_flops_linear(self):

        d = []

        self._get_flops_linear(self._model, d)

        return d

    def _replace_linear(
        self,
        module: torch.nn.Module,
        include_names: List[str] = None,
        output_limit: int = None,
        flops_limit: int = None,
    ):
        for name, child in module.named_children():
            # Check if the current child is an nn.Linear instance
            if isinstance(child, nn.Linear):
                
                include = []

                if include_names is not None:
                    if name in include_names:
                        logging.info(f"Layer nn.Linear {name} is in the list of permitted names.")
                        include.append(True)

                if output_limit is not None:
                    if child.weight.shape[0] > output_limit:
                        include.append(True)
                        logging.info(f"Layer nn.Linear {name} has an output above the limit.")
                        
                if flops_limit is not None:
                    out_features, in_features = child.weight.size()
                    flops = in_features * out_features
                    if flops > flops_limit:
                        include.append(True)
                        logging.info(f"Layer nn.Linear {name} has flops above the limit.")
                        print("added")

                if not any(include):
                    continue

                # Create a new LinearWCkp layer with the same parameters as the original child
                new_child = LinearWCkp(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                )

                # Move the new child to the same device and dtype as the original child
                new_child.to(child.weight.device, child.weight.dtype)

                # Copy the state_dict of the original child to the new child
                new_child.load_state_dict(child.state_dict())

                # Replace the original child with the new child in the module
                setattr(module, name, new_child)

            # Recursively replace linear layers in the current child's children
            self._replace_linear(
                module=child,
                include_names=include_names,
                output_limit=output_limit,
                flops_limit=flops_limit
            )