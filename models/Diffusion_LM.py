import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from einops import rearrange

@dataclass
class DiffusionLMConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    
    # Diffusion specific parameters
    num_diffusion_steps: int = 1000
    noise_schedule: str = "cosine"  # "linear", "cosine", "sqrt"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Model architecture
    time_embedding_dim: int = 128
    use_self_conditioning: bool = True
    
    # Training parameters
    predict_x0: bool = False  # Predict x_0 directly vs predict noise
    loss_type: str = "mse"  # "mse", "l1", "huber"
    
    bias: bool = True

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        """
        time: (batch_size,) tensor of timestep values
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with time conditioning"""
    
    def __init__(self, config: DiffusionLMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim, bias=config.bias)
        self.attn = nn.MultiheadAttention(
            config.dim, config.num_heads, 
            dropout=config.dropout, 
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.dim, bias=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, 4 * config.dim, bias=config.bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.dim, config.dim, bias=config.bias),
            nn.Dropout(config.dropout)
        )
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim)
        )
        
    def forward(self, x, time_emb, attn_mask=None):
        """
        x: (batch, seq_len, dim)
        time_emb: (batch, time_embedding_dim)
        attn_mask: attention mask for causal attention
        """
        # Time conditioning
        time_cond = self.time_mlp(time_emb)  # (batch, dim)
        time_cond = time_cond.unsqueeze(1)  # (batch, 1, dim)
        
        # Self-attention with time conditioning
        residual = x
        x = self.norm1(x + time_cond)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + attn_out
        
        # MLP with time conditioning
        residual = x
        x = self.norm2(x + time_cond)
        x = residual + self.mlp(x)
        
        return x

class NoiseScheduler:
    """Noise scheduler for diffusion process"""
    
    def __init__(self, config: DiffusionLMConfig):
        self.config = config
        self.num_steps = config.num_diffusion_steps
        
        if config.noise_schedule == "linear":
            self.betas = torch.linspace(config.beta_start, config.beta_end, config.num_diffusion_steps)
        elif config.noise_schedule == "cosine":
            self.betas = self.cosine_beta_schedule(config.num_diffusion_steps)
        elif config.noise_schedule == "sqrt":
            self.betas = self.sqrt_beta_schedule(config.num_diffusion_steps, config.beta_start, config.beta_end)
        else:
            raise ValueError(f"Unknown noise schedule: {config.noise_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule from Improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def sqrt_beta_schedule(self, timesteps, beta_start, beta_end):
        """Square root noise schedule"""
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def extract(a, t, x_shape):
    """Extract values from a 1-D tensor for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DiffusionLanguageModel(nn.Module):
    """
    Diffusion Language Model for discrete text generation
    
    References:
    - Diffusion-LM Improves Controllable Text Generation (Li et al., 2022)
    - Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning (Chen et al., 2022)
    """
    
    def __init__(self, config: DiffusionLMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings - map discrete tokens to continuous space
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.dim)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(config.time_embedding_dim)
        
        # Diffusion backbone
        self.layers = nn.ModuleList([
            ResidualBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output layers
        self.norm_out = nn.LayerNorm(config.dim, bias=config.bias)
        
        if config.predict_x0:
            # Predict x_0 (original embeddings) directly
            self.output_projection = nn.Linear(config.dim, config.dim)
        else:
            # Predict noise
            self.output_projection = nn.Linear(config.dim, config.dim)
        
        # Self-conditioning projection (optional)
        if config.use_self_conditioning:
            self.self_cond_projection = nn.Linear(config.dim, config.dim)
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(config)
        
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
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.pos_embedding.weight.numel()
        return n_params
    
    def forward(self, x_t, t, x_self_cond=None, input_ids=None):
        """
        Forward pass of the diffusion model
        
        x_t: (batch, seq_len, dim) - noisy embeddings at timestep t
        t: (batch,) - timestep values
        x_self_cond: (batch, seq_len, dim) - self-conditioning input (optional)
        input_ids: (batch, seq_len) - original token ids (for loss computation)
        """
        batch_size, seq_len, _ = x_t.shape
        device = x_t.device
        
        # Time embedding
        time_emb = self.time_embedding(t)  # (batch, time_embedding_dim)
        
        # Positional embeddings
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embedding(pos)  # (seq_len, dim)
        
        # Start with noisy input
        x = x_t + pos_emb.unsqueeze(0)  # Add positional embeddings
        
        # Self-conditioning
        if self.config.use_self_conditioning and x_self_cond is not None:
            x = x + self.self_cond_projection(x_self_cond)
        
        # Create causal attention mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, time_emb, attn_mask=causal_mask)
        
        # Output projection
        x = self.norm_out(x)
        output = self.output_projection(x)
        
        return output
    
    def compute_loss(self, input_ids, reduction='mean'):
        """
        Compute the diffusion training loss
        
        input_ids: (batch, seq_len) - ground truth token ids
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Convert tokens to embeddings (x_0)
        x_0 = self.token_embedding(input_ids)  # (batch, seq_len, dim)
        
        # Sample random timesteps
        t = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Get noisy embeddings x_t
        x_t = self.noise_scheduler.q_sample(x_0, t, noise)
        
        # Self-conditioning (50% of the time during training)
        x_self_cond = None
        if self.config.use_self_conditioning and np.random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.forward(x_t, t)
        
        # Predict noise or x_0
        predicted = self.forward(x_t, t, x_self_cond)
        
        # Compute loss
        if self.config.predict_x0:
            target = x_0
        else:
            target = noise
        
        if self.config.loss_type == "mse":
            loss = F.mse_loss(predicted, target, reduction=reduction)
        elif self.config.loss_type == "l1":
            loss = F.l1_loss(predicted, target, reduction=reduction)
        elif self.config.loss_type == "huber":
            loss = F.huber_loss(predicted, target, reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        return {
            'loss': loss,
            'predicted': predicted,
            'target': target,
            'x_t': x_t,
            'x_0': x_0,
            't': t
        }
    
    @torch.no_grad()
    def p_sample(self, x_t, t, x_self_cond=None):
        """
        Sample x_{t-1} from x_t using the learned model
        """
        # Get model prediction
        predicted = self.forward(x_t, t, x_self_cond)
        
        if self.config.predict_x0:
            # Model predicts x_0 directly
            pred_x0 = predicted
            # Compute predicted noise
            alpha_cumprod_t = extract(self.noise_scheduler.alphas_cumprod, t, x_t.shape)
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
            pred_noise = (x_t - sqrt_alpha_cumprod_t * pred_x0) / sqrt_one_minus_alpha_cumprod_t
        else:
            # Model predicts noise
            pred_noise = predicted
            # Compute predicted x_0
            alpha_cumprod_t = extract(self.noise_scheduler.alphas_cumprod, t, x_t.shape)
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
            pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_cumprod_t
        
        # Compute x_{t-1}
        alpha_t = extract(self.noise_scheduler.alphas, t, x_t.shape)
        alpha_cumprod_t = extract(self.noise_scheduler.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t_prev = extract(self.noise_scheduler.alphas_cumprod_prev, t, x_t.shape)
        beta_t = extract(self.noise_scheduler.betas, t, x_t.shape)
        
        # Posterior mean
        pred_x_t_prev_mean = (
            torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1 - alpha_cumprod_t) * pred_x0 +
            torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * x_t
        )
        
        # Add noise if not the final step
        if torch.any(t > 0):
            posterior_variance_t = extract(self.noise_scheduler.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            pred_x_t_prev = pred_x_t_prev_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            pred_x_t_prev = pred_x_t_prev_mean
        
        return pred_x_t_prev, pred_x0
    
    @torch.no_grad()
    def sample(self, shape, return_all_timesteps=False):
        """
        Generate samples using the diffusion process
        
        shape: (batch_size, seq_len, dim)
        """
        device = next(self.parameters()).device
        batch_size, seq_len, dim = shape
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        x_self_cond = None
        all_x = [x] if return_all_timesteps else None
        
        # Reverse diffusion process
        for i in reversed(range(self.config.num_diffusion_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Self-conditioning
            if self.config.use_self_conditioning:
                if x_self_cond is None:
                    x_self_cond = self.forward(x, t)
                else:
                    x_self_cond = 0.5 * x_self_cond + 0.5 * self.forward(x, t)
            
            x, pred_x0 = self.p_sample(x, t, x_self_cond)
            
            if return_all_timesteps:
                all_x.append(x)
        
        if return_all_timesteps:
            return torch.stack(all_x, dim=0)
        else:
            return x
    
    def generate_text(self, seq_len, batch_size=1, temperature=1.0, top_k=None):
        """
        Generate text by sampling embeddings and finding nearest tokens
        """
        # Sample embeddings
        shape = (batch_size, seq_len, self.config.dim)
        sampled_embeddings = self.sample(shape)
        
        # Find nearest tokens in embedding space
        # This is a simplified approach - in practice you might use more sophisticated methods
        with torch.no_grad():
            # Compute distances to all token embeddings
            token_embeddings = self.token_embedding.weight  # (vocab_size, dim)
            
            # Reshape for batch computation
            sampled_flat = sampled_embeddings.view(-1, self.config.dim)  # (batch*seq_len, dim)
            
            # Compute cosine similarity (or use L2 distance)
            similarities = F.cosine_similarity(
                sampled_flat.unsqueeze(1),  # (batch*seq_len, 1, dim)
                token_embeddings.unsqueeze(0),  # (1, vocab_size, dim)
                dim=2
            )  # (batch*seq_len, vocab_size)
            
            # Apply temperature and top-k if specified
            if temperature != 1.0:
                similarities = similarities / temperature
            
            if top_k is not None:
                top_k = min(top_k, similarities.size(-1))
                top_similarities, top_indices = torch.topk(similarities, top_k)
                similarities = torch.full_like(similarities, float('-inf'))
                similarities.scatter_(1, top_indices, top_similarities)
            
            # Sample tokens
            probs = F.softmax(similarities, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Reshape back
            token_ids = token_ids.view(batch_size, seq_len)
        
        return {
            'token_ids': token_ids,
            'embeddings': sampled_embeddings,
            'iterations': self.config.num_diffusion_steps,
            'final_diff': 0.0  # For compatibility
        }

def create_diffusion_variants():
    """Create different sized variants of Diffusion LM for comparison"""
    
    # Tiny Diffusion LM
    tiny_config = DiffusionLMConfig(
        vocab_size=100,  # Will be set during training
        max_seq_len=256,
        dim=128,
        num_layers=4,
        num_heads=4,
        num_diffusion_steps=100,
        time_embedding_dim=64
    )
    
    # Small Diffusion LM
    small_config = DiffusionLMConfig(
        vocab_size=100,  # Will be set during training
        max_seq_len=512,
        dim=256,
        num_layers=6,
        num_heads=8,
        num_diffusion_steps=500,
        time_embedding_dim=128
    )
    
    # Medium Diffusion LM
    medium_config = DiffusionLMConfig(
        vocab_size=100,  # Will be set during training
        max_seq_len=1024,
        dim=512,
        num_layers=8,
        num_heads=8,
        num_diffusion_steps=1000,
        time_embedding_dim=256
    )
    
    return {
        'tiny': tiny_config,
        'small': small_config,
        'medium': medium_config
    }

if __name__ == "__main__":
    # Test the model
    config = DiffusionLMConfig(vocab_size=1000, max_seq_len=64, dim=128, num_layers=4, num_diffusion_steps=100)
    model = DiffusionLanguageModel(config)
    
    print(f"Diffusion LM parameters: {model.get_num_params():,}")
    
    # Test forward pass and loss computation
    x = torch.randint(0, 1000, (2, 32))
    
    loss_dict = model.compute_loss(x)
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        generated_dict = model.generate_text(seq_len=20, batch_size=1)
        print(f"Generated tokens shape: {generated_dict['token_ids'].shape}")
        print(f"Diffusion steps: {generated_dict['iterations']}")