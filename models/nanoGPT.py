import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with flash attention support"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.dim % config.num_heads == 0
        
        self.num_heads = config.num_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.num_heads

        
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.dim, 3 * config.dim, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (dim)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.dim, dim=2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Use Flash Attention if available, otherwise manual implementation
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Flash attention (PyTorch 2.0+)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=True
            )
        else:
            # Manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """Multi-layer perceptron used in transformer blocks"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, 4 * config.dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.dim, config.dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block with self-attention and MLP"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoGPT(nn.Module):
    """
    GPT Language Model implementation based on nanoGPT
    References:
    1) the official GPT-2 TensorFlow implementation released by OpenAI:
    https://github.com/openai/gpt-2/blob/master/src/model.py
    2) huggingface/transformers PyTorch implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_seq_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.dim),           # token embeddings
            wpe = nn.Embedding(config.max_seq_len, config.dim),          # position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.dim, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight sharing between token embeddings and output layer (like GPT-2)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        """Initialize weights following GPT-2 paper"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Cannot forward sequence of length {t}, max seq length is only {self.config.max_seq_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, dim)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, dim)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return {
            'logits': logits,
            'loss': loss,
            'iterations': 1,  # For compatibility with iterative model
            'final_diff': 0.0  # For compatibility with iterative model
        }

    def crop_max_seq_len(self, max_seq_len):
        """
        Crop the model's maximum sequence length.
        This is useful for fine-tuning on shorter sequences.
        """
        # cropping max sequence length
        assert max_seq_len <= self.config.max_seq_len
        self.config.max_seq_len = max_seq_len
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:max_seq_len])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:max_seq_len,:max_seq_len]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at max_seq_len
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            # Forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)['logits']
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Optionally apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold (top_p)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Apply the mask to the original logits
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def create_model_variants():
    """Create different sized variants of nanoGPT for comparison"""
    
    # Tiny model (similar to your iterative thinking model)
    tiny_config = GPTConfig(
        vocab_size=100,  # Will be set properly during training
        max_seq_len=256,
        dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1
    )
    
    # Small model
    small_config = GPTConfig(
        vocab_size=100,  # Will be set properly during training
        max_seq_len=512,
        dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    )
    
    # Medium model (closer to GPT-2 small)
    medium_config = GPTConfig(
        vocab_size=100,  # Will be set properly during training
        max_seq_len=1024,
        dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1
    )
    
    return {
        'tiny': tiny_config,
        'small': small_config,
        'medium': medium_config
    }

def estimate_model_flops(model: NanoGPT, seq_len: int):
    """Estimate FLOPs for model forward pass"""
    config = model.config
    N = config.num_layers
    d = config.dim
    h = config.num_heads
    V = config.vocab_size
    T = seq_len
    
    # Embedding lookups (negligible)
    embedding_flops = 0
    
    # Attention layers
    # Q, K, V projections: 3 * T * d * d
    # Attention computation: T * T * d (for each head, then sum over heads)
    # Output projection: T * d * d
    attention_flops_per_layer = 3 * T * d * d + h * T * T * (d // h) + T * d * d
    
    # MLP layers
    # Two linear layers: T * d * 4d + T * 4d * d = 8 * T * d * d
    mlp_flops_per_layer = 8 * T * d * d
    
    # Total transformer flops
    transformer_flops = N * (attention_flops_per_layer + mlp_flops_per_layer)
    
    # Output projection
    output_flops = T * d * V
    
    total_flops = transformer_flops + output_flops
    return total_flops

if __name__ == "__main__":
    # Test the model
    config = GPTConfig(vocab_size=1000, max_seq_len=256, dim=128, num_layers=4, num_heads=4)
    model = NanoGPT(config)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
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