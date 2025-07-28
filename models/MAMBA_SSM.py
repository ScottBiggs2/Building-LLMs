import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from einops import rearrange, repeat

@dataclass
class MambaConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dim: int = 768
    num_layers: int = 12
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4   # Local convolution width
    expand: int = 2   # Block expansion factor
    dt_rank: int = 32  # Rank of Δ (timescale parameter)
    dt_scale: float = 1.0
    dt_init: str = "random"  # "constant" or "random"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True  # Use parallel scan
    dropout: float = 0.1

class SelectiveScan(nn.Module):
    """Core selective scan operation - the heart of Mamba"""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.d_state = config.d_state
        self.pscan = config.pscan
    
    def forward(self, u, delta, A, B, C, D=None):
        """
        Selective scan operation
        u: (batch, seq_len, d_inner)
        delta: (batch, seq_len, d_inner)
        A: (d_inner, d_state)
        B: (batch, seq_len, d_state) 
        C: (batch, seq_len, d_state)
        D: (d_inner,) - skip connection
        """
        batch, seq_len, d_inner = u.shape
        
        # Discretize A and B
        deltaA = torch.exp(rearrange(delta, "b l d -> b l d 1") * rearrange(A, "d n -> 1 1 d n"))
        deltaB_u = rearrange(delta, "b l d -> b l d 1") * rearrange(B, "b l n -> b l 1 n") * rearrange(u, "b l d -> b l d 1")
        
        if self.pscan:
            # Use parallel scan (more efficient for training)
            x = self.parallel_scan(deltaA, deltaB_u)
        else:
            # Sequential scan (easier to understand)
            x = self.sequential_scan(deltaA, deltaB_u)
        
        # Output projection
        y = torch.sum(x * rearrange(C, "b l n -> b l 1 n"), dim=3)
        
        # Skip connection
        if D is not None:
            y = y + u * rearrange(D, "d -> 1 1 d")
        
        return y
    
    def sequential_scan(self, A, Bu):
        """Sequential scan implementation"""
        batch, seq_len, d_inner, d_state = A.shape
        
        # Initialize hidden state
        x = torch.zeros(batch, d_inner, d_state, device=A.device, dtype=A.dtype)
        xs = []
        
        for i in range(seq_len):
            x = A[:, i] * x + Bu[:, i]
            xs.append(x)
        
        return torch.stack(xs, dim=1)  # (batch, seq_len, d_inner, d_state)
    
    def parallel_scan(self, A, Bu):
        """Parallel scan using associative scan"""
        # This is a simplified version - in practice you'd use a more optimized implementation
        return self.sequential_scan(A, Bu)  # Fallback for now

class MambaBlock(nn.Module):
    """Single Mamba block with selective state space model"""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_inner = int(config.expand * config.dim)
        
        # Input projections
        self.in_proj = nn.Linear(config.dim, self.d_inner * 2, bias=config.bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )
        
        # x_proj projects the convolved input to delta, B, C parameters
        self.x_proj = nn.Linear(self.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        
        # dt_proj projects from dt_rank to d_inner (timescale parameter)
        self.dt_proj = nn.Linear(config.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt_proj to output small values
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias to be between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # A parameter - represents the dynamics matrix
        A = repeat(torch.arange(1, config.d_state + 1), "n -> d n", d=self.d_inner).float()
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter - skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, config.dim, bias=config.bias)
        
        # Selective scan
        self.selective_scan = SelectiveScan(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape
        
        # Save residual
        residual = x
        
        # Input projection
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # 1D Convolution
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seq_len]  # Causal conv
        x = rearrange(x, "b d l -> b l d")
        
        # Activation
        x = F.silu(x)
        
        # SSM parameters from x
        x_dbl = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        
        # Compute delta (timescale parameter)
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt + self.dt_proj.bias)  # Ensure positive
        
        # A matrix (dynamics)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Selective scan
        y = self.selective_scan(x, dt, A, B, C, self.D)
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)
        
        # Residual connection
        return output + residual

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class Mamba(nn.Module):
    """
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    
    References:
    - Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
    - Structured State Spaces for Sequence Modeling (Gu et al., 2022)
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final norm
        self.norm_f = RMSNorm(config.dim)
        
        # Language model head
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie embeddings and output weights
        self.lm_head.weight = self.embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.weight.numel()
        return n_params
    
    def forward(self, input_ids, targets=None):
        """
        input_ids: (batch, seq_len)
        targets: (batch, seq_len) - for training
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embeddings(input_ids)  # (batch, seq_len, dim)
        
        # Pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm_f(x)
        
        # Language model head
        if targets is not None:
            # Training mode - compute loss over all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode - only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return {
            'logits': logits,
            'loss': loss,
            'iterations': 1,  # For compatibility
            'final_diff': 0.0  # For compatibility
        }
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate tokens autoregressively
        """
        for _ in range(max_new_tokens):
            # Crop context if too long
            input_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits = self(input_cond)['logits']
            
            # Get last token logits and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

def create_mamba_variants():
    """Create different sized variants of Mamba for comparison"""
    
    # Tiny Mamba
    tiny_config = MambaConfig(
        vocab_size=100,  # Will be set during training
        max_seq_len=256,
        dim=128,
        num_layers=4,
        d_state=8,
        d_conv=4,
        expand=2,
        dt_rank=16
    )
    
    # Small Mamba
    small_config = MambaConfig(
        vocab_size=100,  # Will be set during training
        max_seq_len=512,
        dim=256,
        num_layers=6,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank=32
    )
    
    # Medium Mamba
    medium_config = MambaConfig(
        vocab_size=100,  # Will be set during training
        max_seq_len=1024,
        dim=512,
        num_layers=8,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank=64
    )
    
    return {
        'tiny': tiny_config,
        'small': small_config,
        'medium': medium_config
    }

def estimate_mamba_flops(model: Mamba, seq_len: int):
    """Estimate FLOPs for Mamba forward pass"""
    config = model.config
    N = config.num_layers
    d = config.dim
    d_inner = int(config.expand * d)
    d_state = config.d_state
    d_conv = config.d_conv
    V = config.vocab_size
    T = seq_len
    
    # Embedding (negligible)
    embedding_flops = 0
    
    # Per layer:
    # 1. Input projection: T * d * (2 * d_inner)
    # 2. Conv1D: T * d_inner * d_conv (roughly)
    # 3. x_proj: T * d_inner * (dt_rank + 2*d_state)
    # 4. dt_proj: T * dt_rank * d_inner
    # 5. Selective scan: T * d_inner * d_state (dominant operation)
    # 6. Output projection: T * d_inner * d
    
    flops_per_layer = (
        T * d * (2 * d_inner) +  # Input projection
        T * d_inner * d_conv +   # Conv1D
        T * d_inner * (config.dt_rank + 2 * d_state) +  # x_proj
        T * config.dt_rank * d_inner +  # dt_proj
        T * d_inner * d_state +  # Selective scan (simplified)
        T * d_inner * d  # Output projection
    )
    
    # Total transformer flops
    transformer_flops = N * flops_per_layer
    
    # Output projection
    output_flops = T * d * V
    
    total_flops = transformer_flops + output_flops
    return total_flops

if __name__ == "__main__":
    # Test the model
    config = MambaConfig(vocab_size=1000, max_seq_len=256, dim=128, num_layers=4)
    model = Mamba(config)
    
    print(f"Mamba parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randint(0, 1000, (2, 64))
    y = torch.randint(0, 1000, (2, 64))
    
    output = model(x, y)
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        generated = model.generate(x[:1, :10], max_new_tokens=20, temperature=0.8)
        print(f"Generated sequence shape: {generated.shape}")
    
    # Estimate FLOPs
    flops = estimate_mamba_flops(model, 64)
    print(f"Estimated FLOPs for seq_len=64: {flops:,}")