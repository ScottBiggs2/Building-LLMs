import torch
import torch.nn.functional as F
import pickle
import os
from models.nanoGPT import NanoGPT, GPTConfig
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class NanoGPTInferenceConfig:
    checkpoint_path: str = "checkpoints_nanogpt/nanogpt_small_best.pt"
    data_dir: str = "shakespeare_data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generation parameters
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    
    # Model compilation
    compile_model: bool = False

class NanoGPTInference:
    def __init__(self, config: NanoGPTInferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"Loading NanoGPT model on {self.device}...")
        
        # Load vocabulary
        self.load_vocabulary()
        
        # Load model
        self.model = self.load_model()
        
        print("NanoGPT model loaded successfully!")
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
        self.vocab_size = vocab_data['vocab_size']
        
    def load_model(self):
        """Load the trained NanoGPT model"""
        if not os.path.exists(self.config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # Extract model config from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            gpt_config = GPTConfig(
                vocab_size=model_config.vocab_size,
                max_seq_len=model_config.max_seq_len,
                dim=model_config.dim,
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                dropout=0.0,  # No dropout during inference
                bias=model_config.bias
            )
        else:
            # Fallback: try to infer from model state dict
            print("Warning: No config found in checkpoint, using default parameters")
            gpt_config = GPTConfig(
                vocab_size=self.vocab_size,
                max_seq_len=256,
                dim=256,
                num_layers=6,
                num_heads=8,
                dropout=0.0,
                bias=True
            )
        
        # Create model
        model = NanoGPT(gpt_config)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Compile model if requested and available
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
        
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
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, top_k: Optional[int] = None,
                top_p: Optional[float] = None, verbose: bool = True) -> dict:
        """Generate text using autoregressive generation"""
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        
        if verbose:
            print(f"Generating with prompt: '{prompt}'")
            print(f"Max new tokens: {max_new_tokens}, Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
        
        # Encode prompt
        prompt_ids = self.encode_text(prompt)
        input_tensor = torch.tensor([prompt_ids], device=self.device)
        
        # Generation timing
        start_time = time.time()
        
        with torch.no_grad():
            # Use model's built-in generation method with top-p support
            generated = self.generate_with_top_p(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Decode result
        generated_ids = generated[0].tolist()
        generated_text = self.decode_tokens(generated_ids)
        
        # Extract just the generated part
        prompt_text = self.decode_tokens(prompt_ids)
        if len(generated_text) > len(prompt_text):
            generated_part = generated_text[len(prompt_text):]
        else:
            generated_part = ""
        
        if verbose:
            tokens_per_second = max_new_tokens / generation_time if generation_time > 0 else 0
            print(f"Generation completed in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
        
        return {
            'text': generated_text,
            'prompt': prompt_text,
            'generated_part': generated_part,
            'generation_time': generation_time,
            'tokens_per_second': max_new_tokens / generation_time if generation_time > 0 else 0,
            'iterations': 1,  # For compatibility with iterative model
            'final_diff': 0.0  # For compatibility with iterative model
        }
    
    def generate_with_top_p(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Enhanced generation method with top-p (nucleus) sampling support
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at max_seq_len
            idx_cond = idx if idx.size(1) <= self.model.config.max_seq_len else idx[:, -self.model.config.max_seq_len:]
            
            # Forward the model to get the logits for the index in the sequence
            with torch.no_grad():
                logits = self.model(idx_cond)['logits']
            
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
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("NANOGPT SHAKESPEARE MODEL - CHAT MODE")
        print("="*60)
        print("Enter your prompts and the model will complete them using autoregressive generation!")
        print("Commands:")
        print("  /temp <value>  - Set temperature (0.1-2.0)")
        print("  /tokens <int>  - Set max new tokens")
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
                    elif command.startswith('/tokens '):
                        try:
                            tokens = int(command.split()[1])
                            self.config.max_new_tokens = max(10, min(500, tokens))
                            print(f"Max new tokens set to {self.config.max_new_tokens}")
                        except:
                            print("Usage: /tokens <int>")
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
                result = self.generate(user_input, verbose=verbose)
                
                # Display results
                print(f"\nGenerated text:")
                print(f"'{result['text']}'")
                
                if verbose:
                    print(f"\nGeneration details:")
                    print(f"  Time: {result['generation_time']:.2f}s")
                    print(f"  Speed: {result['tokens_per_second']:.1f} tokens/s")
                    print(f"  Generated part: '{result['generated_part']}'")
                
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type /quit to exit or continue...")
                continue
            except Exception as e:
                print(f"Error during generation: {e}")
                continue
    
    def benchmark_speed(self, prompt: str = "HAMLET:", num_runs: int = 5, max_new_tokens: int = 100):
        """Benchmark generation speed"""
        print(f"Benchmarking NanoGPT speed with {num_runs} runs...")
        
        times = []
        for i in range(num_runs):
            result = self.generate(
                prompt, 
                max_new_tokens=max_new_tokens,
                verbose=False
            )
            times.append(result['generation_time'])
            print(f"Run {i+1}: {result['generation_time']:.2f}s ({result['tokens_per_second']:.1f} tokens/s)")
        
        avg_time = sum(times) / len(times)
        avg_speed = max_new_tokens / avg_time
        
        print(f"\nBenchmark Results:")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Average speed: {avg_speed:.1f} tokens/s")
        print(f"  Min time: {min(times):.2f}s")
        print(f"  Max time: {max(times):.2f}s")
        
        return {
            'avg_time': avg_time,
            'avg_speed': avg_speed,
            'times': times
        }

def main():
    """Main inference script"""
    config = NanoGPTInferenceConfig(
        # Update these paths based on your trained model
        checkpoint_path="checkpoints_nanogpt/nanogpt_small_best.pt",
        compile_model=False  # Set to True if you have PyTorch 2.0+
    )
    
    try:
        # Initialize inference
        inference = NanoGPTInference(config)
        
        # Quick test generation
        print("\nTesting model with a quick generation...")
        test_result = inference.generate("HAMLET:", max_new_tokens=50, verbose=True)
        print(f"Test result: '{test_result['text']}'")
        
        # Benchmark speed
        print("\nRunning speed benchmark...")
        benchmark_results = inference.benchmark_speed()
        
        # Start interactive chat
        inference.chat_loop()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Trained model at the specified checkpoint path")
        print("2. Vocabulary file at: shakespeare_data/vocab.pkl")
        print("3. Run the training script first to create the model")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()