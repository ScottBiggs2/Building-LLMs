import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time

from models.Diffusion_LM import DiffusionLanguageModel, DiffusionLMConfig
from data.data_class import ShakespeareDataset

class DiffusionDataset(Dataset):
    """Dataset for diffusion language model training"""
    
    def __init__(self, data_path: str, seq_len: int, stride: int = None):
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
        
        # Load encoded data
        self.data = np.load(data_path)
        
        # Create sliding windows
        self.samples = []
        for i in range(0, len(self.data) - seq_len, self.stride):
            self.samples.append(self.data[i:i + seq_len])
        
        print(f"Created {len(self.samples)} diffusion samples with seq_len={seq_len}, stride={self.stride}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.tensor(sample, dtype=torch.long)

@dataclass
class DiffusionLMTrainingConfig:
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
    
    # Diffusion specific
    num_diffusion_steps: int = 500
    noise_schedule: str = "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    time_embedding_dim: int = 128
    use_self_conditioning: bool = True
    predict_x0: bool = False
    loss_type: str = "mse"
    
    # Training
    batch_size: int = 16  # Smaller batch for diffusion models
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_epochs: int = 100  # More epochs needed for diffusion
    
    # Learning rate schedule
    warmup_steps: int = 2000
    lr_decay_steps: int = None
    min_lr: float = 3e-5
    
    # Evaluation
    eval_interval: int = 1000  # Less frequent due to slow generation
    eval_steps: int = 50
    generation_interval: int = 2000  # Generate samples less frequently
    
    # Logging and saving
    log_interval: int = 100
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints_diffusion"
    use_wandb: bool = False
    
    # Device and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

class DiffusionLMTrainer:
    """Trainer for Diffusion Language Model"""
    
    def __init__(self, config: DiffusionLMTrainingConfig):
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
                project="diffusion-lm-shakespeare",
                config=config.__dict__,
                name=f"diffusion-{config.model_size}-{config.dim}d-{config.num_layers}L-{config.num_diffusion_steps}steps"
            )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.training_metrics = {
            'losses': [],
            'learning_rates': [],
            'grad_norms': [],
            'timestep_losses': []  # Track loss by timestep
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
            'tiny': {
                'dim': 128, 'num_layers': 4, 'num_heads': 4, 'max_seq_len': 128,
                'num_diffusion_steps': 100, 'time_embedding_dim': 64
            },
            'small': {
                'dim': 256, 'num_layers': 6, 'num_heads': 8, 'max_seq_len': 256,
                'num_diffusion_steps': 500, 'time_embedding_dim': 128
            },
            'medium': {
                'dim': 512, 'num_layers': 8, 'num_heads': 8, 'max_seq_len': 512,
                'num_diffusion_steps': 1000, 'time_embedding_dim': 256
            }
        }
        
        if self.config.model_size in architectures:
            arch = architectures[self.config.model_size]
            for key, value in arch.items():
                setattr(self.config, key, value)
            
            print(f"Using {self.config.model_size} Diffusion LM architecture:")
            print(f"  Dimension: {self.config.dim}")
            print(f"  Layers: {self.config.num_layers}")
            print(f"  Heads: {self.config.num_heads}")
            print(f"  Max seq len: {self.config.max_seq_len}")
            print(f"  Diffusion steps: {self.config.num_diffusion_steps}")
            print(f"  Time embedding dim: {self.config.time_embedding_dim}")
    
    def create_model(self):
        """Create the Diffusion Language Model"""
        diffusion_config = DiffusionLMConfig(
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
            dim=self.config.dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            num_diffusion_steps=self.config.num_diffusion_steps,
            noise_schedule=self.config.noise_schedule,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            time_embedding_dim=self.config.time_embedding_dim,
            use_self_conditioning=self.config.use_self_conditioning,
            predict_x0=self.config.predict_x0,
            loss_type=self.config.loss_type
        )
        
        model = DiffusionLanguageModel(diffusion_config)
        model = model.to(self.device)
        
        # Compile model if available and requested
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        non_embedding_params = model.get_num_params(non_embedding=True)
        
        print(f"Diffusion LM created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-embedding parameters: {non_embedding_params:,}")
        print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB")
        
        return model
    
    def create_dataloaders(self):
        """Create training and validation dataloaders"""
        train_dataset = DiffusionDataset(
            os.path.join(self.config.data_dir, "train_encoded.npy"),
            seq_len=self.config.max_seq_len
        )
        
        val_dataset = DiffusionDataset(
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
        """Create optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2)
        )
    
    def create_scheduler(self):
        """Create learning rate scheduler"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            elif step < self.config.lr_decay_steps:
                progress = (step - self.config.warmup_steps) / (self.config.lr_decay_steps - self.config.warmup_steps)
                import math
                return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - self.config.min_lr / self.config.learning_rate) + self.config.min_lr / self.config.learning_rate
            else:
                return self.config.min_lr / self.config.learning_rate
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: torch.Tensor) -> Dict:
        """Single training step for diffusion model"""
        input_ids = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Compute diffusion loss
        loss_dict = self.model.compute_loss(input_ids)
        loss = loss_dict['loss']
        
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
        
        # Track timestep-specific losses for analysis
        timesteps = loss_dict['t'].cpu().numpy()
        for t in timesteps:
            if len(self.training_metrics['timestep_losses']) <= t:
                self.training_metrics['timestep_losses'].extend([[] for _ in range(t + 1 - len(self.training_metrics['timestep_losses']))])
            self.training_metrics['timestep_losses'][t].append(loss.item())
        
        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'timesteps': timesteps
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
                
            input_ids = batch.to(self.device)
            loss_dict = self.model.compute_loss(input_ids)
            total_loss += loss_dict['loss'].item()
            num_batches += 1
        
        self.model.train()
        
        return {
            'val_loss': total_loss / num_batches if num_batches > 0 else float('inf')
        }
    
    def generate_sample(self, seq_len: int = 100, temperature: float = 1.0) -> str:
        """Generate text sample using diffusion process"""
        self.model.eval()
        
        with torch.no_grad():
            # Generate sample
            start_time = time.time()
            generated_dict = self.model.generate_text(
                seq_len=seq_len,
                batch_size=1,
                temperature=temperature
            )
            generation_time = time.time() - start_time
            
            # Decode to text
            token_ids = generated_dict['token_ids'][0].cpu().tolist()
            generated_text = ''.join([self.idx_to_char.get(i, '?') for i in token_ids])
        
        self.model.train()
        
        return generated_text, generation_time, generated_dict['iterations']
    
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
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"diffusion_{self.config.model_size}_step_{self.step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, f"diffusion_{self.config.model_size}_best.pt")
            torch.save(checkpoint, best_path)
            
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("Starting Diffusion LM training...")
        print(f"Device: {self.device}")
        print(f"Total steps per epoch: {len(self.train_loader)}")
        print(f"Total training steps: {len(self.train_loader) * self.config.max_epochs}")
        print(f"Diffusion steps: {self.config.num_diffusion_steps}")
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            
            # Training loop
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                train_dict = self.train_step(batch)
                self.step += 1
                
                # Update progress bar
                avg_timestep = np.mean(train_dict['timesteps'])
                progress_bar.set_postfix({
                    'loss': f"{train_dict['loss'].item():.4f}",
                    'lr': f"{train_dict['lr']:.2e}",
                    'avg_t': f"{avg_timestep:.0f}"
                })
                
                # Logging
                if self.step % self.config.log_interval == 0:
                    if self.config.use_wandb:
                        wandb.log({
                            'train/loss': train_dict['loss'].item(),
                            'train/lr': train_dict['lr'],
                            'train/grad_norm': train_dict['grad_norm'],
                            'train/avg_timestep': avg_timestep,
                            'step': self.step
                        })
                
                # Evaluation
                if self.step % self.config.eval_interval == 0:
                    val_metrics = self.evaluate()
                    
                    print(f"\nStep {self.step} - Val Loss: {val_metrics['val_loss']:.4f}")
                    
                    if self.config.use_wandb:
                        wandb.log({
                            'val/loss': val_metrics['val_loss'],
                            'step': self.step
                        })
                    
                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint(is_best=True)
                
                # Generate sample (less frequent due to computational cost)
                if self.step % self.config.generation_interval == 0:
                    print("Generating sample text...")
                    sample_text, gen_time, diffusion_steps = self.generate_sample(seq_len=100)
                    print(f"Sample generation ({gen_time:.1f}s, {diffusion_steps} steps):")
                    print(f"{sample_text[:200]}...\n")
                    
                    if self.config.use_wandb:
                        wandb.log({
                            'sample/text': wandb.Html(f"<pre>{sample_text}</pre>"),
                            'sample/generation_time': gen_time,
                            'sample/diffusion_steps': diffusion_steps,
                            'step': self.step
                        })
                
                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()
    
    def plot_training_curves(self):
        """Plot training metrics including timestep analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
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
        axes[0, 2].plot(self.training_metrics['grad_norms'])
        axes[0, 2].set_title('Gradient Norm')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Grad Norm')
        
        # Loss by timestep
        if self.training_metrics['timestep_losses']:
            timestep_avg_losses = []
            timesteps = []
            for t, losses in enumerate(self.training_metrics['timestep_losses']):
                if losses:
                    timestep_avg_losses.append(np.mean(losses))
                    timesteps.append(t)
            
            if timestep_avg_losses:
                axes[1, 0].plot(timesteps, timestep_avg_losses)
                axes[1, 0].set_title('Loss by Timestep')
                axes[1, 0].set_xlabel('Timestep')
                axes[1, 0].set_ylabel('Average Loss')
        
        # Loss (log scale)
        axes[1, 1].semilogy(self.training_metrics['losses'])
        axes[1, 1].set_title('Training Loss (Log Scale)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        
        # Timestep distribution histogram
        if self.training_metrics['timestep_losses']:
            all_timesteps = []
            for t, losses in enumerate(self.training_metrics['timestep_losses']):
                all_timesteps.extend([t] * len(losses))
            
            if all_timesteps:
                axes[1, 2].hist(all_timesteps, bins=50, alpha=0.7)
                axes[1, 2].set_title('Timestep Distribution')
                axes[1, 2].set_xlabel('Timestep')
                axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.checkpoint_dir, 'diffusion_training_curves.png'))
        plt.show()

def main():
    """Main training script"""
    # Configuration
    config = DiffusionLMTrainingConfig(
        data_dir="shakespeare_data",
        model_size="tiny",  # Start with tiny for testing
        batch_size=8,  # Small batch size for diffusion
        learning_rate=3e-4,
        max_epochs=30,
        use_wandb=False,
        compile_model=False,
        warmup_steps=1000
    )
    
    # Create trainer
    trainer = DiffusionLMTrainer(config)
    
    # Test model first
    print("Testing Diffusion LM with a small batch...")
    sample_input = torch.randint(0, config.vocab_size, (2, 32)).to(config.device)
    
    with torch.no_grad():
        test_loss_dict = trainer.model.compute_loss(sample_input)
    print(f"Test loss: {test_loss_dict['loss'].item():.4f}")
    
    # Test generation
    print("Testing generation...")
    sample_text, gen_time, steps = trainer.generate_sample(seq_len=50)
    print(f"Sample generation ({gen_time:.1f}s): {sample_text[:100]}...")
    
    # Start training
    trainer.train()
    
    # Plot results
    trainer.plot_training_curves()

if __name__ == "__main__":
    main()

# python -m training.Diffusion_LM_trainer