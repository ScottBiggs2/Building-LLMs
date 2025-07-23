import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LearnedWaveletActivation(nn.Module):
    """Custom activation with learned wavelet-like behavior and exponential decay"""
    def __init__(self, dim, num_wavelets=4):
        super().__init__()
        self.dim = dim
        self.num_wavelets = num_wavelets
        
        # Learnable parameters for wavelets
        self.frequencies = nn.Parameter(torch.randn(num_wavelets, dim) * 0.1)
        self.phases = nn.Parameter(torch.randn(num_wavelets, dim) * 0.1)
        self.amplitudes = nn.Parameter(torch.ones(num_wavelets, dim) * 0.5)
        
        # Exponential decay parameters
        self.decay_rate = nn.Parameter(torch.ones(dim) * 0.1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        
        # Create position encoding for wavelet computation
        pos = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
        
        # Compute wavelet components
        wavelet_sum = torch.zeros_like(x)
        for i in range(self.num_wavelets):
            freq = self.frequencies[i].unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
            phase = self.phases[i].unsqueeze(0).unsqueeze(0)
            amp = self.amplitudes[i].unsqueeze(0).unsqueeze(0)
            
            wavelet = amp * torch.sin(freq * pos + phase)
            wavelet_sum += wavelet
        
        # Apply exponential decay to prevent blowups
        decay = torch.exp(-self.decay_rate.abs().unsqueeze(0).unsqueeze(0) * torch.abs(x))
        
        return x * (1 + wavelet_sum) * decay

class TransformerLayer(nn.Module):
    """Standard transformer layer for intermediate processing"""
    def __init__(self, dim, ff_dim, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            LearnedWaveletActivation(ff_dim),
            nn.Linear(ff_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class IterativeThinkingLLM(nn.Module):
    """
    LLM with network-level iterative thinking:
    x -> [concat with h_j] -> L0 -> L1 -> ... -> L_{n-1} -> [h_i, h_j]
    h_j gets fed back for next iteration until convergence
    """
    def __init__(self, vocab_size, dim=512, num_layers=6, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # L0: Input layer (like tokenizer)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        
        # Project concatenated input + h_j to proper dimension
        self.input_projection = nn.Linear(dim * 2, dim)  # concat x and h_j
        
        # L1 to L_{n-2}: Intermediate layers (standard transformer layers)
        self.intermediate_layers = nn.ModuleList([
            TransformerLayer(dim, dim * 4) 
            for _ in range(num_layers - 2)
        ])
        
        # L_{n-1}: Final layer with custom activation that outputs h_i and h_j
        self.final_layer = nn.Sequential(
            TransformerLayer(dim, dim * 4),
            LearnedWaveletActivation(dim)
        )
        
        # Split final layer output into h_i (for output) and h_j (for recurrence)
        self.output_head = nn.Linear(dim, vocab_size)  # h_i -> logits
        self.recurrence_head = nn.Linear(dim, dim)     # h_j for next iteration
        
        # Convergence parameters - don't like that these are hardcoded args but it's fine for now I guess
        self.convergence_threshold = 1e-4
        self.max_iterations = 200
        self.min_iterations = 3
        
    def get_initial_embedding(self, input_ids):
        """L0: Convert tokens to embeddings"""
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(pos_ids)
        
        return token_emb + pos_emb
    
    def forward_pass(self, x, h_j_prev):
        """Single forward pass through the entire network"""
        batch_size, seq_len, dim = x.shape
        
        # Concatenate input with previous h_j (or zeros for first iteration)
        if h_j_prev is None:
            h_j_prev = torch.zeros_like(x)
        
        # Concatenate along feature dimension
        concatenated = torch.cat([x, h_j_prev], dim=-1)  # (batch, seq_len, 2*dim)
        
        # Project back to original dimension
        hidden = self.input_projection(concatenated)  # (batch, seq_len, dim)
        
        # Pass through intermediate layers L1 to L_{n-2}
        for layer in self.intermediate_layers:
            hidden = layer(hidden)
        
        # Final layer L_{n-1} with custom activation
        final_hidden = self.final_layer(hidden)
        
        # Split into h_i and h_j
        h_i = final_hidden  # Used for output
        h_j = self.recurrence_head(final_hidden)  # Fed back for next iteration
        
        return h_i, h_j
    
    def check_convergence(self, h_j_current, h_j_previous):
        """Check if h_j has converged"""
        if h_j_previous is None:
            return False, float('inf')
        
        diff = torch.abs(h_j_current - h_j_previous).mean().item()
        return diff <= self.convergence_threshold, diff
    
    def detect_periodicity(self, history, window=3):
        """Simple periodicity detection"""
        if len(history) < 2 * window:
            return False
            
        recent = history[-window:]
        prev = history[-2*window:-window]
        
        similarity = sum(abs(a - b) for a, b in zip(recent, prev)) / window
        return similarity < self.convergence_threshold 
    
    def forward(self, input_ids):
        """
        Forward pass with iterative thinking:
        1. Get initial embeddings x
        2. Initialize h_j = None (zeros)
        3. Repeat: x + h_j -> network -> h_i, h_j_new
        4. Continue until h_j converges
        5. Output final h_i
        """
        # L0: Get initial embeddings
        x = self.get_initial_embedding(input_ids)
        
        # Initialize recurrent state
        h_j = None
        h_j_history = []
        diff_history = []
        
        # Iterative thinking loop
        for iteration in range(self.max_iterations):
            # Single forward pass through entire network
            h_i, h_j_new = self.forward_pass(x, h_j)
            
            # Check convergence of h_j
            converged, diff = self.check_convergence(h_j_new, h_j)
            diff_history.append(diff)
            h_j_history.append(h_j_new.clone() if h_j_new is not None else None)
            
            # Update h_j for next iteration
            h_j = h_j_new
            
            if iteration >= self.min_iterations:
                # Check for convergence
                if converged:
                    print(f"h_j converged at iteration {iteration + 1}, diff: {diff:.6f}")
                    break
                    
                # Check for periodicity
                if self.detect_periodicity(diff_history):
                    print(f"Detected periodic behavior at iteration {iteration + 1}")
                    break
                    
                # Check for exponential decay
                if len(diff_history) >= 3: # only start checking at 3
                    recent_diffs = diff_history[-5:]
                    decay_ok = all(recent_diffs[i] < recent_diffs[i-1] * 0.95 for i in range(1, 3))
                    # Only allow stopping if mean diff is below a reasonable threshold
                    if decay_ok and np.mean(recent_diffs) < 1e-2:
                        print(f"Detected exponential decay at iteration {iteration + 1} (mean diff {np.mean(recent_diffs):.4e})")
                        break

                # Check if unsuccessful:
                if len(diff_history) >= self.max_iterations-2:
                    print(f"Exceeded maximum thinking iterations")
                    break
        
        # Convert final h_i to output logits
        logits = self.output_head(h_i)
        
        return {
            'logits': logits,
            'iterations': iteration + 1,
            'final_diff': diff_history[-1] if diff_history else 0,
            'convergence_history': diff_history,
            'final_h_i': h_i,
            'final_h_j': h_j
        }
