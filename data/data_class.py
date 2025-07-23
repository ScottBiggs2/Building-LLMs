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