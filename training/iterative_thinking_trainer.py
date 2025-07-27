import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pickle
import os
import gc  # Import the garbage collector
import warnings  # <-- Add for warning suppression
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# Import the iterative thinking model (assuming it's in iterative_thinking_llm.py)
from models.iterative_thinking_LLM import IterativeThinkingLLM
from data.data_class import ShakespeareDataset, NonAutoregressiveDataset

@dataclass
class TrainingConfig:
    # Data
    data_dir: str = "shakespeare_data"
    
    # Model
    vocab_size: int = None  # Will be loaded from data
    dim: int = 256
    num_layers: int = 6
    max_seq_len: int = 256
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5  # Minimum learning rate for cosine decay
    weight_decay: float = 0.1
    max_epochs: int = 100
    warmup_steps: int = 100
    lr_decay_steps: int = None  # Set automatically based on epochs and dataset size
    gradient_accumulation_steps: int = 4  # Key for memory saving
    grad_clip: float = 1.0
    
    # Evaluation
    eval_interval: int = 500
    eval_steps: int = 100
    
    # Convergence-specific
    convergence_weight: float = 0.05  # Weight for convergence loss
    efficiency_weight: float = 0.01  # Penalty for too many iterations
    min_acceptable_iterations: int = 8
    max_acceptable_iterations: int = 32
    
    # Logging
    log_interval: int = 100
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints_iterative"
    use_wandb: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DataLoader and memory management
    num_workers: int = 0  # Default to 0 for safety
    pin_memory: bool = False  # Default to False for safety
    max_samples: Optional[int] = None  # Limit number of samples loaded

class IterativeThinkingTrainer:
    """Trainer for the Iterative Thinking LLM"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Load vocabulary
        self.load_vocabulary()
        
        # Create model
        self.model = self.create_model()
        
        # Create datasets and dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        # Create optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Set learning rate schedule parameters
        if self.config.lr_decay_steps is None:
            self.config.lr_decay_steps = self.config.max_epochs * (len(self.train_loader) // self.config.gradient_accumulation_steps)
        
        self.scheduler = self.create_scheduler()
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project="iterative-thinking-llm",
                config=config.__dict__,
                name=f"shakespeare-dim{config.dim}-layers{config.num_layers}"
            )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.training_metrics = {
            'losses': [],
            'iteration_counts': [],
            'convergence_diffs': [],
            'learning_rates': []
        }
    
    def load_vocabulary(self):
        """Load vocabulary from processed data"""
        vocab_path = os.path.join(self.config.data_dir, "vocab.pkl")
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        # Reserve the last token as a MASK token for non-autoregressive generation
        self.config.vocab_size = vocab_data['vocab_size'] + 1  # +1 for MASK token
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = vocab_data['idx_to_char']
        
        # Add MASK token
        self.char_to_idx['<MASK>'] = self.config.vocab_size - 1
        self.idx_to_char[self.config.vocab_size - 1] = '<MASK>'
        
        print(f"Loaded vocabulary: {vocab_data['vocab_size']} characters + 1 MASK token = {self.config.vocab_size} total")
    
    def create_model(self):
        """Create the iterative thinking model"""
        # from paste import IterativeThinkingLLM  # Import from the provided file
        
        model = IterativeThinkingLLM(
            vocab_size=self.config.vocab_size,
            dim=self.config.dim,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len
        )
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB")
        
        return model
    
    def create_dataloaders(self):
        """Create training and validation dataloaders for non-autoregressive training"""
        train_dataset = NonAutoregressiveDataset(
            os.path.join(self.config.data_dir, "train_encoded.npy"),
            seq_len=self.config.max_seq_len,
            mask_ratio=0.6,  # Mask 60% of tokens for training
            max_samples=self.config.max_samples
        )
        
        val_dataset = NonAutoregressiveDataset(
            os.path.join(self.config.data_dir, "val_encoded.npy"),
            seq_len=self.config.max_seq_len,
            mask_ratio=0.6,
            max_samples=self.config.max_samples
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
        
        print(f"Created non-autoregressive dataloaders:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        
        return train_loader, val_loader
    
    def compute_loss(self, output: Dict, targets: torch.Tensor, mask_positions: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute multi-component loss for non-autoregressive iterative thinking"""
        logits = output['logits']
        iterations = output['iterations']
        final_diff = output['final_diff']
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Primary language modeling loss
        lm_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=-1
        )
        
        # Optional: Focused loss only on masked positions
        if mask_positions is not None and len(mask_positions) > 0:
            mask_loss = 0.0
            for batch_idx in range(batch_size):
                if len(mask_positions[batch_idx]) > 0:
                    batch_logits = logits[batch_idx][mask_positions[batch_idx]]
                    batch_targets = targets[batch_idx][mask_positions[batch_idx]]
                    mask_loss += F.cross_entropy(batch_logits, batch_targets)
            
            if batch_size > 0:
                mask_loss = mask_loss / batch_size
                lm_loss = 0.7 * lm_loss + 0.3 * mask_loss
        
        # Convergence loss - encourage actual convergence
        convergence_loss = torch.tensor(final_diff, device=self.device) * self.config.convergence_weight
        
        # Iteration efficiency loss - encourage a "healthy" number of iterations
        efficiency_loss = torch.tensor(0.0, device=self.device)
        
        # Only apply efficiency penalty after the learning rate warmup period
        if self.step > self.config.warmup_steps:
            min_iter = self.config.min_acceptable_iterations
            max_iter = self.config.max_acceptable_iterations
            
            # Penalize stopping too early (quadratic penalty)
            if iterations < min_iter:
                penalty = ((min_iter - iterations) / min_iter) ** 2
                efficiency_loss = penalty * self.config.efficiency_weight
            # Penalize thinking for too long (quadratic penalty)
            elif iterations > max_iter:
                penalty = ((iterations - max_iter) / max_iter) ** 2
                efficiency_loss = penalty * self.config.efficiency_weight
        
        total_loss = lm_loss + convergence_loss + efficiency_loss
        
        return {
            'total': total_loss,
            'lm': lm_loss,
            'convergence': convergence_loss,
            'efficiency': efficiency_loss,
            'iterations': iterations,
            'final_diff': final_diff
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate model on validation set using non-autoregressive approach"""
        self.model.eval()
        
        total_loss = 0
        total_iterations = 0
        total_convergence_diff = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            if num_batches >= self.config.eval_steps:
                break
            try:
                inputs, targets, mask_positions = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Replace -1 placeholders with actual mask token
                mask_token_id = self.config.vocab_size - 1
                inputs = torch.where(inputs == -1, mask_token_id, inputs)
                output = self.model(inputs)
                loss_dict = self.compute_loss(output, targets, mask_positions)
                total_loss += loss_dict['lm'].item()
                total_iterations += loss_dict['iterations']
                total_convergence_diff += loss_dict['final_diff']
                num_batches += 1
                # Memory management
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("[WARNING] Out of memory during evaluation. Skipping batch.")
                    self.clear_memory()
                    continue
                else:
                    raise
        self.model.train()
        if num_batches == 0:
            return {'val_loss': float('inf'), 'avg_iterations': 0, 'avg_convergence_diff': 0}
        return {
            'val_loss': total_loss / num_batches,
            'avg_iterations': total_iterations / num_batches,
            'avg_convergence_diff': total_convergence_diff / num_batches
        }
    
    def create_scheduler(self):
        """Create learning rate scheduler with warmup and cosine decay."""
        def lr_lambda(current_step: int):
            # Linear warmup
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            
            # Cosine decay
            progress = float(current_step - self.config.warmup_steps) / float(max(1, self.config.lr_decay_steps - self.config.warmup_steps))
            # Clamp progress to [0, 1]
            progress = max(0.0, min(1.0, progress))
            
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed_lr = (1 - self.config.min_lr / self.config.learning_rate) * cosine_decay + self.config.min_lr / self.config.learning_rate
            return decayed_lr

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def generate_sample(self, prompt: str = "HAMLET:", target_length: int = 200) -> str:
        """Generate text sample using non-autoregressive parallel generation"""
        self.model.eval()
        
        # Encode prompt and pad to target length
        prompt_ids = [self.char_to_idx.get(c, 0) for c in prompt]
        
        # Create input sequence: prompt + special tokens for positions to generate
        # We'll use a special "MASK" token (vocab_size-1) for positions to generate
        mask_token = self.config.vocab_size - 1
        
        # Prepare input: [prompt] + [MASK] * (target_length - len(prompt))
        total_length = min(target_length, self.config.max_seq_len)
        if len(prompt_ids) >= total_length:
            input_ids = prompt_ids[:total_length]
        else:
            input_ids = prompt_ids + [mask_token] * (total_length - len(prompt_ids))
        
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            # Single forward pass generates ALL tokens simultaneously
            output = self.model(input_tensor)
            logits = output['logits'][0]  # Shape: (seq_len, vocab_size)
            
            # Sample all tokens at once from the logits
            probs = F.softmax(logits, dim=-1)
            generated_tokens = torch.multinomial(probs, 1).squeeze(-1)  # Shape: (seq_len,)
            
            # Keep the original prompt, only replace the masked positions
            final_tokens = input_ids.copy()
            prompt_len = len(prompt_ids)
            
            # Replace masked positions with generated tokens
            for i in range(prompt_len, len(final_tokens)):
                final_tokens[i] = generated_tokens[i].item()
        
        # Decode generated text
        generated_text = ''.join([self.idx_to_char.get(i, '?') for i in final_tokens])
        
        self.model.train()
        return generated_text, output['iterations'], output['final_diff']
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,  # Save as a dictionary for portability
            'training_metrics': self.training_metrics,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{self.step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def clear_memory(self):
        """Aggressively clear memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()

    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Total steps per epoch: {len(self.train_loader)}")
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            
            # Training loop
            self.model.train()
            self.optimizer.zero_grad()
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    inputs, targets, mask_positions = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    mask_token_id = self.config.vocab_size - 1
                    inputs = torch.where(inputs == -1, mask_token_id, inputs)

                    # Forward pass
                    output = self.model(inputs)
                    loss_dict = self.compute_loss(output, targets, mask_positions)
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = loss_dict['total'] / self.config.gradient_accumulation_steps
                    scaled_loss.backward()

                    progress_bar.set_postfix({
                        'loss': f"{loss_dict['lm'].item():.4f}",
                        'iter': loss_dict['iterations'],
                        'conv': f"{loss_dict['final_diff']:.1e}"
                    })

                    # Perform optimizer step after accumulating gradients
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.step += 1

                        # Update metrics for plotting at every actual step
                        self.training_metrics['losses'].append(loss_dict['lm'].item())
                        self.training_metrics['iteration_counts'].append(loss_dict['iterations'])
                        self.training_metrics['convergence_diffs'].append(loss_dict['final_diff'])
                        self.training_metrics['learning_rates'].append(self.scheduler.get_last_lr()[0])

                        # Logging
                        if self.step % self.config.log_interval == 0:
                            if self.config.use_wandb:
                                wandb.log({
                                    'train/loss': loss_dict['lm'].item(),
                                    'train/convergence_loss': loss_dict['convergence'].item(),
                                    'train/efficiency_loss': loss_dict['efficiency'].item(),
                                    'train/iterations': loss_dict['iterations'],
                                    'train/convergence_diff': loss_dict['final_diff'],
                                    'train/lr': self.scheduler.get_last_lr()[0],
                                    'step': self.step
                                })

                        # Evaluation & Checkpointing (tied to optimizer steps)
                        if self.step % self.config.eval_interval == 0:
                            val_metrics = self.evaluate()
                            print(f"\nStep {self.step} - Val Loss: {val_metrics['val_loss']:.4f}, Avg Iterations: {val_metrics['avg_iterations']:.1f}")
                            sample_text, sample_iterations, sample_convergence = self.generate_sample()
                            print(f"Sample (iters: {sample_iterations}, conv: {sample_convergence:.1e}): {sample_text[:200]}...\n")

                            if self.config.use_wandb:
                                wandb.log({
                                    'val/loss': val_metrics['val_loss'], 'val/avg_iterations': val_metrics['avg_iterations'],
                                    'val/avg_convergence_diff': val_metrics['avg_convergence_diff'],
                                    'sample/text': wandb.Html(f"<pre>{sample_text}</pre>"),
                                    'sample/iterations': sample_iterations, 'sample/convergence_diff': sample_convergence,
                                    'step': self.step
                                })

                            if val_metrics['val_loss'] < self.best_val_loss:
                                self.best_val_loss = val_metrics['val_loss']
                                self.save_checkpoint(is_best=True)

                        if self.step % self.config.save_interval == 0:
                            self.save_checkpoint()

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"\n[WARNING] OOM on batch {batch_idx}. Skipping optimizer step and clearing memory.")
                        self.clear_memory()
                        self.optimizer.zero_grad() # Crucial: clear partial gradients
                        continue
                    else:
                        raise
        
        # Always save final model at end of training
        self.save_checkpoint(is_best=False)
    
    def plot_training_curves(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curve
        axes[0, 0].plot(self.training_metrics['losses'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        
        # Iteration count
        axes[0, 1].plot(self.training_metrics['iteration_counts'])
        axes[0, 1].set_title('Iterations per Forward Pass')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Iterations')
        
        # Convergence diff
        axes[1, 0].semilogy(self.training_metrics['convergence_diffs'])
        axes[1, 0].set_title('Final Convergence Difference')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Difference (log scale)')
        
        # Learning rate
        axes[1, 1].plot(self.training_metrics['learning_rates'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('LR')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.checkpoint_dir, 'training_curves.png'))
        plt.show()

def main():
    """Main training script"""
    # Configuration - ready for a large overnight run
    config = TrainingConfig(
        data_dir="shakespeare_data",
        dim=256,
        num_layers=6,
        max_seq_len=256,
        batch_size=32,  # Start smaller for this complex model
        gradient_accumulation_steps=1, # Effective batch size = 32 * 4 = 128
        learning_rate=1e-4,
        max_epochs = 1,
        use_wandb=False,  # Set to True if you want to use wandb
        num_workers=0,  # Safe default
        pin_memory=False,  # Safe default
        max_samples=10000  # Limit for debugging/low memory
    )
    
    # Create trainer
    trainer = IterativeThinkingTrainer(config)
    
    # Test model first
    print("Testing model with a small batch...")
    sample_input = torch.randint(0, config.vocab_size, (2, 32)).to(config.device)
    with torch.no_grad():
        test_output = trainer.model(sample_input)
    print(f"Test output shape: {test_output['logits'].shape}")
    print(f"Test iterations: {test_output['iterations']}")
    
    # Start training
    trainer.train()
    
    # Plot results
    trainer.plot_training_curves()

if __name__ == "__main__":
    main()


# python -m training.iterative_thinking_trainer