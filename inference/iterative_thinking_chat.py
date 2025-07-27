import torch
import torch.nn.functional as F
import pickle
import os
from models.iterative_thinking_LLM import IterativeThinkingLLM
# Import TrainingConfig to allow torch.load to unpickle the object from the checkpoint
from training.iterative_thinking_trainer import TrainingConfig
from dataclasses import dataclass
from typing import Optional

@dataclass
class InferenceConfig:
    checkpoint_path: str = "checkpoints_iterative/checkpoint_step_313.pt"
    data_dir: str = "shakespeare_data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generation parameters
    max_length: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

class IterativeThinkingInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"Loading model on {self.device}...")
        
        # Load vocabulary
        self.load_vocabulary()
        
        # Load model
        self.model = self.load_model()
        
        print("Model loaded successfully!")
        print(f"Vocabulary size: {len(self.char_to_idx)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def load_vocabulary(self):
        """Load vocabulary from processed data"""
        vocab_path = os.path.join(self.config.data_dir, "vocab.pkl")
        
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = vocab_data['idx_to_char']
        
        # Add MASK token (should match training setup)
        vocab_size = vocab_data['vocab_size'] + 1
        self.char_to_idx['<MASK>'] = vocab_size - 1
        self.idx_to_char[vocab_size - 1] = '<MASK>'
        
        self.vocab_size = vocab_size
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # Extract model config from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            # Handle both dict (from newer checkpoints) and dataclass object (from older checkpoints)
            if isinstance(model_config, dict):
                model = IterativeThinkingLLM(
                    vocab_size=model_config['vocab_size'],
                    dim=model_config['dim'],
                    num_layers=model_config['num_layers'],
                    max_seq_len=model_config['max_seq_len']
                )
            else:  # Assume it's a dataclass object
                model = IterativeThinkingLLM(
                    vocab_size=model_config.vocab_size,
                    dim=model_config.dim,
                    num_layers=model_config.num_layers,
                    max_seq_len=model_config.max_seq_len
                )
        else:
            # Fallback: try to infer from model state dict
            print("Warning: No config found in checkpoint, using default parameters")
            model = IterativeThinkingLLM(
                vocab_size=self.vocab_size,
                dim=256,
                num_layers=6,
                max_seq_len=256
            )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Print training info if available
        if 'step' in checkpoint:
            print(f"Loaded model from step {checkpoint['step']}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        
        return model
    
    def encode_text(self, text: str) -> list:
        """Encode text to token IDs"""
        return [self.char_to_idx.get(c, 0) for c in text]
    
    def decode_tokens(self, token_ids: list) -> str:
        """Decode token IDs to text"""
        return ''.join([self.idx_to_char.get(i, '?') for i in token_ids])
    
    def apply_sampling(self, logits: torch.Tensor, temperature: float = 1.0, 
                      top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        """Apply temperature, top-k, and top-p sampling"""
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(1, top_k_indices, top_k_logits)
            logits = logits_filtered
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def generate(self, prompt: str, max_length: Optional[int] = None, 
                temperature: Optional[float] = None, top_k: Optional[int] = None,
                top_p: Optional[float] = None, verbose: bool = True) -> dict:
        """Generate text using non-autoregressive parallel generation"""
        # Use config defaults if not specified
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        
        if verbose:
            print(f"Generating with prompt: '{prompt}'")
            print(f"Max length: {max_length}, Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
        
        # Encode prompt
        prompt_ids = self.encode_text(prompt)
        mask_token_id = self.vocab_size - 1
        
        # Create input sequence: prompt + MASK tokens
        total_length = min(max_length, self.model.max_seq_len)
        if len(prompt_ids) >= total_length:
            input_ids = prompt_ids[:total_length]
            mask_start = len(input_ids)  # No generation needed
        else:
            input_ids = prompt_ids + [mask_token_id] * (total_length - len(prompt_ids))
            mask_start = len(prompt_ids)
        
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            # Single forward pass with iterative thinking
            if verbose:
                print("Model is thinking...")
            
            output = self.model(input_tensor)
            logits = output['logits'][0]  # Shape: (seq_len, vocab_size)
            
            if verbose:
                print(f"Thinking completed in {output['iterations']} iterations")
                print(f"Final convergence difference: {output['final_diff']:.2e}")
            
            # Non-autoregressive generation: apply sampling to all positions at once
            sampled_tokens = prompt_ids.copy()
            
            # Get logits for the positions to be generated
            generation_logits = logits[mask_start:]
            
            if generation_logits.shape[0] > 0:
                # Apply sampling (temp, top-k, top-p) to all generation logits in parallel
                sampled_logits = self.apply_sampling(generation_logits, temperature, top_k, top_p)
                
                # Get probabilities and sample new tokens
                probs = F.softmax(sampled_logits, dim=-1)
                new_tokens = torch.multinomial(probs, 1).squeeze(-1).tolist()
                sampled_tokens.extend(new_tokens)

        # Decode result
        generated_text = self.decode_tokens(sampled_tokens)
        
        return {
            'text': generated_text,
            'prompt': prompt,
            'generated_part': generated_text[len(prompt):],
            'iterations': output['iterations'],
            'convergence_diff': output['final_diff'],
            'thinking_metadata': {
                'convergence_history': output.get('convergence_history', []),
                'final_h_i_norm': torch.norm(output['final_h_i']).item() if 'final_h_i' in output else None,
                'final_h_j_norm': torch.norm(output['final_h_j']).item() if 'final_h_j' in output else None
            }
        }
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("ITERATIVE THINKING SHAKESPEARE MODEL - CHAT MODE")
        print("="*60)
        print("Enter your prompts and the model will complete them using iterative thinking!")
        print("Commands:")
        print("  /temp <value>  - Set temperature (0.1-2.0)")
        print("  /length <int>  - Set max generation length")
        print("  /topk <int>    - Set top-k sampling")
        print("  /topp <float>  - Set top-p sampling")
        print("  /verbose       - Toggle verbose output")
        print("  /quit          - Exit")
        print("-" * 60)
        
        verbose = True
        
        while True:
            try:
                user_input = input("\nPrompt: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower()
                    
                    if command == '/quit':
                        print("Goodbye!")
                        break
                    elif command == '/verbose':
                        verbose = not verbose
                        print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                        continue
                    elif command.startswith('/temp '):
                        try:
                            temp = float(command.split()[1])
                            self.config.temperature = max(0.1, min(2.0, temp))
                            print(f"Temperature set to {self.config.temperature}")
                        except:
                            print("Usage: /temp <float>")
                        continue
                    elif command.startswith('/length '):
                        try:
                            length = int(command.split()[1])
                            self.config.max_length = max(10, min(500, length))
                            print(f"Max length set to {self.config.max_length}")
                        except:
                            print("Usage: /length <int>")
                        continue
                    elif command.startswith('/topk '):
                        try:
                            topk = int(command.split()[1])
                            self.config.top_k = max(1, min(100, topk))
                            print(f"Top-k set to {self.config.top_k}")
                        except:
                            print("Usage: /topk <int>")
                        continue
                    elif command.startswith('/topp '):
                        try:
                            topp = float(command.split()[1])
                            self.config.top_p = max(0.1, min(1.0, topp))
                            print(f"Top-p set to {self.config.top_p}")
                        except:
                            print("Usage: /topp <float>")
                        continue
                    else:
                        print("Unknown command. Type /quit to exit.")
                        continue
                
                # Generate response
                print("\n" + "-" * 40)
                start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                
                if start_time:
                    start_time.record()
                
                result = self.generate(user_input, verbose=verbose)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    generation_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                else:
                    generation_time = None
                
                # Display results
                print(f"\nGenerated text:")
                print(f"'{result['text']}'")
                
                if verbose:
                    print(f"\nGeneration details:")
                    print(f"  Iterations: {result['iterations']}")
                    print(f"  Convergence: {result['convergence_diff']:.2e}")
                    if generation_time:
                        print(f"  Time: {generation_time:.2f}s")
                    print(f"  Generated part: '{result['generated_part']}'")
                
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type /quit to exit or continue...")
                continue
            except Exception as e:
                print(f"Error during generation: {e}")
                continue

def main():
    """Main inference script"""
    config = InferenceConfig()
    
    try:
        # Initialize inference
        inference = IterativeThinkingInference(config)
        
        # Quick test generation
        print("\nTesting model with a quick generation...")
        test_result = inference.generate("HAMLET:", max_length=100, verbose=True)
        print(f"Test result: '{test_result['text']}'")
        
        # Start interactive chat
        inference.chat_loop()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Trained model at: checkpoints_iterative/my_iterative_thinking_model.pth")
        print("2. Vocabulary file at: shakespeare_data/vocab.pkl")
        print("3. Model files in the correct directories")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# python -m inference.iterative_thinking_chat