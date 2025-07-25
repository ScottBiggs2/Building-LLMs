import torch
import torch.nn.functional as F
import numpy as np
import pickle
import math
import os
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time

from models.nanoGPT import NanoGPT, GPTConfig
from data.data_class import ShakespeareDataset

@dataclass
class NanoGPTTrainingConfig:
    # Data
    data_dir: str = "shakespeare_data"
    
    # Model size ('tiny', 'small', 'medium')
    model_size: str = "small"
    
    # Model architecture (will be set based on model_size)
    vocab_size: int = None  # Will be loaded from data
    dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 256
    dropout: float = 0.1
    bias: bool = True
    
    # Training
    batch_size: int = 32
    learning_rate: float = 6e-4  # GPT-2 style learning rate
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_epochs: int = 50
    
    # Learning rate schedule
    warmup_steps: int = 2000
    lr_decay_steps: int = None  # Will be set to max_steps
    min_lr: float = 6e-5  # Minimum learning rate
    
    # Evaluation
    eval_interval: int = 500
    eval_steps: int = 100
    
    # Logging and saving
    log_interval: int = 100
    save_interval: int = 2000
    checkpoint_dir: str = "checkpoints_nanogpt"
    use_wandb: bool = False
    
    # Device and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = True  # Use torch.compile if available
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

class NanoGPTTrainer:
    """Trainer for NanoGPT model"""
    
    def __init__(self, config: NanoGPTTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Load vocabulary
        self.load_vocabulary()
        
        # Set model architecture based on size
        self.set_model_architecture()
        
        # Create model
        self.model = self.create_model()
        
        # Create datasets and dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        # Set learning rate schedule parameters
        if self.config.lr_decay_steps is None:
            self.config.lr_decay_steps = len(self.train_loader) * self.config.max_epochs
        
        # Create optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project="nanogpt-shakespeare",
                config=config.__dict__,
                name=f"nanogpt-{config.model_size}-{config.dim}d-{config.num_layers}L"
            )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.training_metrics = {
            'losses': [],
            'learning_rates': [],
            'grad_norms': []
        }
    
    def load_vocabulary(self):
        """Load vocabulary from processed data"""
        vocab_path = os.path.join(self.config.data_dir, "vocab.pkl")
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.config.vocab_size = vocab_data['vocab_size']
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = vocab_data['idx_to_char']
        
        print(f"Loaded vocabulary: {self.config.vocab_size} characters")
    
    def set_model_architecture(self):
        """Set model architecture based on size configuration"""
        architectures = {
            'tiny': {'dim': 128, 'num_layers': 4, 'num_heads': 4, 'max_seq_len': 256},
            'small': {'dim': 256, 'num_layers': 6, 'num_heads': 8, 'max_seq_len': 512},
            'medium': {'dim': 512, 'num_layers': 8, 'num_heads': 8, 'max_seq_len': 1024}
        }
        
        if self.config.model_size in architectures:
            arch = architectures[self.config.model_size]
            self.config.dim = arch['dim']
            self.config.num_layers = arch['num_layers']
            self.config.num_heads = arch['num_heads']
            self.config.max_seq_len = arch['max_seq_len']
            
            print(f"Using {self.config.model_size} architecture:")
            print(f"  Dimension: {self.config.dim}")
            print(f"  Layers: {self.config.num_layers}")
            print(f"  Heads: {self.config.num_heads}")
            print(f"  Max seq len: {self.config.max_seq_len}")
    
    def create_model(self):
        """Create the NanoGPT model"""
        gpt_config = GPTConfig(
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
            dim=self.config.dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            bias=self.config.bias
        )
        
        model = NanoGPT(gpt_config)
        model = model.to(self.device)
        
        # Compile model if available and requested
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        non_embedding_params = model.get_num_params(non_embedding=True)
        
        print(f"Model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-embedding parameters: {non_embedding_params:,}")
        print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB")
        
        return model
    
    def create_dataloaders(self):
        """Create training and validation dataloaders"""
        train_dataset = ShakespeareDataset(
            os.path.join(self.config.data_dir, "train_encoded.npy"),
            seq_len=self.config.max_seq_len
        )
        
        val_dataset = ShakespeareDataset(
            os.path.join(self.config.data_dir, "val_encoded.npy"),
            seq_len=self.config.max_seq_len
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        print(f"Created dataloaders:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        
        return train_loader, val_loader
    
    def create_optimizer(self):
        """Create optimizer with weight decay configuration like GPT-2"""
        # Separate parameters that should and shouldn't be weight decayed
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                
                if pn.endswith('bias'):
                    # All biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # Weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # Weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        
        # Special case the position embeddings
        no_decay.add('transformer.wpe.weight')
        
        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        
        # Create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        return optimizer
    
    def create_scheduler(self):
        """Create learning rate scheduler with warmup and cosine decay"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Linear warmup
                return step / self.config.warmup_steps
            elif step < self.config.lr_decay_steps:
                # Cosine decay
                progress = (step - self.config.warmup_steps) / (self.config.lr_decay_steps - self.config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - self.config.min_lr / self.config.learning_rate) + self.config.min_lr / self.config.learning_rate
            else:
                # Minimum learning rate
                return self.config.min_lr / self.config.learning_rate
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """Single training step"""
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(inputs, targets)
        loss = output['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update metrics
        self.training_metrics['losses'].append(loss.item())
        self.training_metrics['learning_rates'].append(self.scheduler.get_last_lr()[0])
        self.training_metrics['grad_norms'].append(grad_norm.item())
        
        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            if num_batches >= self.config.eval_steps:
                break
                
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            output = self.model(inputs, targets)
            total_loss += output['loss'].item()
            num_batches += 1
        
        self.model.train()
        
        return {
            'val_loss': total_loss / num_batches if num_batches > 0 else float('inf')
        }
    
    def generate_sample(self, prompt: str = "HAMLET:", max_new_tokens: int = 200, 
                       temperature: float = 0.8, top_k: int = 50) -> str:
        """Generate text sample using autoregressive generation"""
        self.model.eval()
        
        # Encode prompt
        prompt_ids = [self.char_to_idx.get(c, 0) for c in prompt]
        input_tensor = torch.tensor([prompt_ids], device=self.device)
        
        with torch.no_grad():
            # Generate using model's built-in generation method
            generated = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode generated text
        generated_ids = generated[0].tolist()
        generated_text = ''.join([self.idx_to_char.get(i, '?') for i in generated_ids])
        
        self.model.train()
        return generated_text
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"nanogpt_{self.config.model_size}_step_{self.step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, f"nanogpt_{self.config.model_size}_best.pt")
            torch.save(checkpoint, best_path)
            
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("Starting NanoGPT training...")
        print(f"Device: {self.device}")
        print(f"Total steps per epoch: {len(self.train_loader)}")
        print(f"Total training steps: {len(self.train_loader) * self.config.max_epochs}")
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            
            # Training loop
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                train_dict = self.train_step(batch)
                self.step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{train_dict['loss'].item():.4f}",
                    'lr': f"{train_dict['lr']:.2e}",
                    'grad_norm': f"{train_dict['grad_norm']:.2f}"
                })
                
                # Logging
                if self.step % self.config.log_interval == 0:
                    if self.config.use_wandb:
                        wandb.log({
                            'train/loss': train_dict['loss'].item(),
                            'train/lr': train_dict['lr'],
                            'train/grad_norm': train_dict['grad_norm'],
                            'step': self.step
                        })
                
                # Evaluation
                if self.step % self.config.eval_interval == 0:
                    val_metrics = self.evaluate()
                    
                    print(f"\nStep {self.step} - Val Loss: {val_metrics['val_loss']:.4f}")
                    
                    # Generate sample
                    sample_text = self.generate_sample()
                    print(f"Sample generation:\n{sample_text[:200]}...\n")
                    
                    if self.config.use_wandb:
                        wandb.log({
                            'val/loss': val_metrics['val_loss'],
                            'sample_text': wandb.Html(f"<pre>{sample_text}</pre>"),
                            'step': self.step
                        })
                    
                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()
    
    def plot_training_curves(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curve
        axes[0, 0].plot(self.training_metrics['losses'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        
        # Learning rate
        axes[0, 1].plot(self.training_metrics['learning_rates'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        
        # Gradient norm
        axes[1, 0].plot(self.training_metrics['grad_norms'])
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Grad Norm')
        
        # Loss (log scale)
        axes[1, 1].semilogy(self.training_metrics['losses'])
        axes[1, 1].set_title('Training Loss (Log Scale)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.checkpoint_dir, 'nanogpt_training_curves.png'))
        plt.show()

def main():
    """Main training script"""
    import math
    
    # Configuration
    config = NanoGPTTrainingConfig(
        data_dir="shakespeare_data",
        model_size="small",  # 'tiny', 'small', 'medium'
        batch_size=16,
        learning_rate=6e-4,
        max_epochs=20,
        use_wandb=False,
        compile_model=False,  # Set to True if you have PyTorch 2.0+
        warmup_steps=1000
    )
    
    # Create trainer
    trainer = NanoGPTTrainer(config)
    
    # Test model first
    print("Testing model with a small batch...")
    sample_input = torch.randint(0, config.vocab_size, (2, 32)).to(config.device)
    sample_target = torch.randint(0, config.vocab_size, (2, 32)).to(config.device)
    
    with torch.no_grad():
        test_output = trainer.model(sample_input, sample_target)
    print(f"Test output logits shape: {test_output['logits'].shape}")
    print(f"Test loss: {test_output['loss'].item():.4f}")
    
    # Start training
    trainer.train()
    
    # Plot results
    trainer.plot_training_curves()

if __name__ == "__main__":
    main()