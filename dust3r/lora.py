import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, 
                 linear_layer: nn.Linear,
                 rank: int = 4,
                 alpha: float = 1.0,
                 dropout: float = 0.0,
                 device=None):
        super().__init__()
        
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(
            torch.zeros((rank, linear_layer.in_features), device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((linear_layer.out_features, rank), device=device)
        )
        self.scaling = alpha / rank
        
        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward
        base_out = self.linear(x)
        
        # LoRA forward
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return base_out + lora_out

def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze all parameters except LoRA parameters"""
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

def add_lora_to_linear_layer(layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> LoRALinear:
    """Helper function to replace a linear layer with a LoRA-augmented version"""
    return LoRALinear(
        linear_layer=layer,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        device=layer.weight.device
    ) 