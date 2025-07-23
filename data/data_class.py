import torch
from torch.utils.data import Dataset
import numpy as np


class ShakespeareDataset(Dataset):
    """Dataset for character-level Shakespeare data with sliding windows"""
    
    def __init__(self, data_path: str, seq_len: int, stride: int = None):
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2  # 50% overlap by default
        
        # Load encoded data
        self.data = np.load(data_path)
        
        # Create sliding windows
        self.samples = []
        for i in range(0, len(self.data) - seq_len, self.stride):
            self.samples.append(self.data[i:i + seq_len + 1])  # +1 for target
        
        print(f"Created {len(self.samples)} samples with seq_len={seq_len}, stride={stride}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.tensor(sample[:-1], dtype=torch.long)  # input
        y = torch.tensor(sample[1:], dtype=torch.long)   # target (shifted by 1)
        return x, y
    

class NonAutoregressiveDataset(Dataset):
    """Dataset for non-autoregressive training with masked sequences"""
    
    def __init__(self, data_path: str, seq_len: int, mask_ratio: float = 0.5, stride: int = None, max_samples: int = None):
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
        self.mask_ratio = mask_ratio  # Fraction of tokens to mask for training
        # Use memory-mapped loading for large files
        self.data = np.load(data_path, mmap_mode='r')
        # Create sliding windows
        self.samples = []
        for i in range(0, len(self.data) - seq_len, self.stride):
            self.samples.append(self.data[i:i + seq_len])
            # Limit number of samples if max_samples is set
            if max_samples is not None and len(self.samples) >= max_samples:
                break
        print(f"Created {len(self.samples)} non-autoregressive samples with seq_len={seq_len}, "
              f"mask_ratio={mask_ratio}, stride={stride}, max_samples={max_samples}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = self.samples[idx].copy()
        target = torch.tensor(sequence, dtype=torch.long)
        
        # Create masked input for non-autoregressive training
        input_seq = sequence.copy()
        
        # Randomly mask tokens (except maybe the first few for context)
        context_len = max(1, int(len(sequence) * 0.1))  # Keep 10% as context
        mask_start = context_len
        
        # Determine which tokens to mask
        maskable_positions = list(range(mask_start, len(sequence)))
        num_to_mask = int(len(maskable_positions) * self.mask_ratio)
        
        if num_to_mask > 0:
            mask_positions = np.random.choice(maskable_positions, num_to_mask, replace=False)
            # Use vocab_size as the mask token (will be handled in vocab loading)
            for pos in mask_positions:
                input_seq[pos] = -1  # Placeholder, will be replaced with actual mask token
        
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        
        return input_tensor, target, torch.tensor(mask_positions if num_to_mask > 0 else [], dtype=torch.long)
