import torch
import torch.nn.functional as F
import pickle
import os
# Import the training config to allow torch.load to unpickle the object from the checkpoint
from training.Diffusion_LM_trainer import DiffusionLMTrainingConfig
from models.Diffusion_LM import DiffusionLanguageModel, DiffusionLMConfig
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np

@dataclass
class DiffusionLMInferenceConfig:
    checkpoint_path: str = "checkpoints_diffusion/diffusion_tiny_5000.pt"
    data_dir: str = "shakespeare_data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generation parameters
    seq_len: int = 200
    temperature: float = 1.0
    top_k: int = 50
    
    # Diffusion-specific parameters
    num_inference_steps: int = None  # Use model's default if None
    guidance_scale: float = 1.0  # For classifier-free guidance (if implemented)
    
    # Model compilation
    compile_model: bool = False

class DiffusionLMInference:
    def __init__(self, config: DiffusionLMInferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"Loading Diffusion LM on {self.device}...")
        
        # Load vocabulary
        self.load_vocabulary()
        
        # Load model
        self.model = self.load_model()
        
        print("Diffusion LM loaded successfully!")
        print(f"Vocabulary size: {len(self.char_to_idx)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Diffusion steps: {self.model.config.num_diffusion_steps}")
        
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
        """Load the trained Diffusion LM"""
        if not os.path.exists(self.config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # Extract model config from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            diffusion_config = DiffusionLMConfig(
                vocab_size=model_config.vocab_size,
                max_seq_len=model_config.max_seq_len,
                dim=model_config.dim,
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                dropout=0.0,  # No dropout during inference
                num_diffusion_steps=model_config.num_diffusion_steps,
                noise_schedule=model_config.noise_schedule,
                beta_start=model_config.beta_start,
                beta_end=model_config.beta_end,
                time_embedding_dim=model_config.time_embedding_dim,
                use_self_conditioning=model_config.use_self_conditioning,
                predict_x0=model_config.predict_x0,
                loss_type=model_config.loss_type
            )
        else:
            # Fallback
            print("Warning: No config found in checkpoint, using default parameters")
            diffusion_config = DiffusionLMConfig(
                vocab_size=self.vocab_size,
                max_seq_len=256,
                dim=256,
                num_layers=6,
                num_heads=8,
                dropout=0.0,
                num_diffusion_steps=500
            )
        
        # Create model
        model = DiffusionLanguageModel(diffusion_config)
        
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
    
    def generate(self, seq_len: Optional[int] = None, temperature: Optional[float] = None,
                top_k: Optional[int] = None, num_inference_steps: Optional[int] = None,
                verbose: bool = True, return_intermediate: bool = False) -> dict:
        """Generate text using diffusion process"""
        # Use config defaults if not specified
        seq_len = seq_len or self.config.seq_len

        # Clamp seq_len to the model's maximum supported sequence length to prevent IndexError
        if seq_len > self.model.config.max_seq_len:
            print(f"Warning: Requested sequence length {seq_len} is greater than model's max {self.model.config.max_seq_len}. Clamping to max.")
            seq_len = self.model.config.max_seq_len

        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        
        # Set inference steps (can be different from training steps for speed)
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps or self.model.config.num_diffusion_steps
        
        if verbose:
            print(f"Generating sequence of length {seq_len}")
            print(f"Temperature: {temperature}, Top-k: {top_k}")
            print(f"Inference steps: {num_inference_steps} (vs {self.model.config.num_diffusion_steps} training steps)")
        
        # Generation timing
        start_time = time.time()
        
        with torch.no_grad():
            if return_intermediate:
                # Generate with intermediate steps for visualization
                generated_dict = self.generate_with_intermediate_steps(seq_len, temperature, top_k, num_inference_steps)
            else:
                # Standard generation
                generated_dict = self.model.generate_text(
                    seq_len=seq_len,
                    batch_size=1,
                    temperature=temperature,
                    top_k=top_k
                )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Decode result
        token_ids = generated_dict['token_ids'][0].cpu().tolist()
        generated_text = self.decode_tokens(token_ids)
        
        if verbose:
            tokens_per_second = seq_len / generation_time if generation_time > 0 else 0
            print(f"Generation completed in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
            print(f"Effective steps per second: {num_inference_steps / generation_time:.1f}")
        
        result = {
            'text': generated_text,
            'token_ids': token_ids,
            'generation_time': generation_time,
            'tokens_per_second': seq_len / generation_time if generation_time > 0 else 0,
            'iterations': generated_dict['iterations'],
            'final_diff': generated_dict['final_diff'],
            'num_inference_steps': num_inference_steps
        }
        
        if return_intermediate:
            result['intermediate_embeddings'] = generated_dict.get('intermediate_embeddings', None)
            result['intermediate_texts'] = generated_dict.get('intermediate_texts', [])
        
        return result
    
    def generate_with_intermediate_steps(self, seq_len: int, temperature: float, top_k: int, num_steps: int):
        """Generate text with intermediate denoising steps saved"""
        shape = (1, seq_len, self.model.config.dim)
        
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        x_self_cond = None
        
        intermediate_embeddings = [x.clone()]
        intermediate_texts = []
        
        # Create custom timestep schedule (can skip steps for faster inference)
        if num_steps < self.model.config.num_diffusion_steps:
            # Use DDIM-style skipping
            step_ratio = self.model.config.num_diffusion_steps // num_steps
            timesteps = list(range(0, self.model.config.num_diffusion_steps, step_ratio))
            timesteps = timesteps[::-1]  # Reverse for denoising
        else:
            timesteps = list(reversed(range(self.model.config.num_diffusion_steps)))
        
        # Reverse diffusion process
        for i, timestep in enumerate(timesteps):
            t = torch.full((1,), timestep, device=self.device, dtype=torch.long)
            
            # Self-conditioning
            if self.model.config.use_self_conditioning:
                if x_self_cond is None:
                    x_self_cond = self.model.forward(x, t)
                else:
                    x_self_cond = 0.5 * x_self_cond + 0.5 * self.model.forward(x, t)
            
            # Denoising step
            x, pred_x0 = self.model.p_sample(x, t, x_self_cond)
            
            # Save intermediate results every few steps
            if i % max(1, len(timesteps) // 10) == 0:
                intermediate_embeddings.append(x.clone())
                
                # Convert current embedding to text for visualization
                with torch.no_grad():
                    # Find nearest tokens
                    token_embeddings = self.model.token_embedding.weight
                    x_flat = x.view(-1, self.model.config.dim)
                    similarities = F.cosine_similarity(
                        x_flat.unsqueeze(1),
                        token_embeddings.unsqueeze(0),
                        dim=2
                    )
                    
                    if top_k is not None:
                        top_k_sim, top_k_indices = torch.topk(similarities, min(top_k, similarities.size(-1)))
                        similarities = torch.full_like(similarities, float('-inf'))
                        similarities.scatter_(1, top_k_indices, top_k_sim)
                    
                    probs = F.softmax(similarities / temperature, dim=-1)
                    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    tokens = tokens.view(1, seq_len)
                    
                    intermediate_text = self.decode_tokens(tokens[0].tolist())
                    intermediate_texts.append(f"Step {i}/{len(timesteps)}: {intermediate_text[:100]}...")
        
        # Final conversion to tokens
        with torch.no_grad():
            token_embeddings = self.model.token_embedding.weight
            x_flat = x.view(-1, self.model.config.dim)
            similarities = F.cosine_similarity(
                x_flat.unsqueeze(1),
                token_embeddings.unsqueeze(0),
                dim=2
            )
            
            if top_k is not None:
                top_k_sim, top_k_indices = torch.topk(similarities, min(top_k, similarities.size(-1)))
                similarities = torch.full_like(similarities, float('-inf'))
                similarities.scatter_(1, top_k_indices, top_k_sim)
            
            probs = F.softmax(similarities / temperature, dim=-1)
            final_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            final_tokens = final_tokens.view(1, seq_len)
        
        return {
            'token_ids': final_tokens,
            'embeddings': x,
            'iterations': len(timesteps),
            'final_diff': 0.0,
            'intermediate_embeddings': intermediate_embeddings,
            'intermediate_texts': intermediate_texts
        }
    
    def guided_generation(self, prompt: str, seq_len: int, guidance_strength: float = 2.0):
        """Generate text with prompt guidance (experimental)"""
        # Clamp seq_len to the model's maximum supported sequence length to prevent IndexError
        if seq_len > self.model.config.max_seq_len:
            print(f"Warning: Requested sequence length {seq_len} is greater than model's max {self.model.config.max_seq_len}. Clamping to max.")
            seq_len = self.model.config.max_seq_len

        print(f"Generating {seq_len} tokens guided by prompt: '{prompt}'")
        
        # Encode prompt
        prompt_ids = self.encode_text(prompt)
        prompt_length = len(prompt_ids)
        
        if prompt_length >= seq_len:
            print("Warning: Prompt longer than target sequence length")
            return self.decode_tokens(prompt_ids[:seq_len])
        
        shape = (1, seq_len, self.model.config.dim)
        
        # Start from noise
        x = torch.randn(shape, device=self.device)
        
        # Set the prompt part to the actual embeddings
        with torch.no_grad():
            prompt_embeddings = self.model.token_embedding(torch.tensor([prompt_ids], device=self.device))
            x[:, :prompt_length, :] = prompt_embeddings[0, :prompt_length, :]
        
        x_self_cond = None
        
        # Reverse diffusion with guidance
        for i in reversed(range(self.model.config.num_diffusion_steps)):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            
            # Self-conditioning
            if self.model.config.use_self_conditioning:
                if x_self_cond is None:
                    x_self_cond = self.model.forward(x, t)
                else:
                    x_self_cond = 0.5 * x_self_cond + 0.5 * self.model.forward(x, t)
            
            # Standard denoising
            x_denoised, pred_x0 = self.model.p_sample(x, t, x_self_cond)
            
            # Apply guidance by keeping prompt part fixed
            if i > self.model.config.num_diffusion_steps // 2:  # Only in early steps
                with torch.no_grad():
                    prompt_embeddings = self.model.token_embedding(torch.tensor([prompt_ids], device=self.device))
                    x_denoised[:, :prompt_length, :] = (
                        guidance_strength * prompt_embeddings[0, :prompt_length, :] +
                        (1 - guidance_strength) * x_denoised[:, :prompt_length, :]
                    )
            
            x = x_denoised
        
        # Convert to tokens
        with torch.no_grad():
            token_embeddings = self.model.token_embedding.weight
            x_flat = x.view(-1, self.model.config.dim)
            similarities = F.cosine_similarity(
                x_flat.unsqueeze(1),
                token_embeddings.unsqueeze(0),
                dim=2
            )
            
            probs = F.softmax(similarities, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            tokens = tokens.view(1, seq_len)
            
            generated_text = self.decode_tokens(tokens[0].tolist())
        
        return generated_text
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("DIFFUSION LM SHAKESPEARE MODEL - CHAT MODE")
        print("="*60)
        print("Enter commands to generate text using diffusion denoising!")
        print("Note: Generation is slow but produces high-quality diverse outputs")
        print("Commands:")
        print("  /generate <length>     - Generate text of specified length")
        print("  /temp <value>          - Set temperature (0.5-2.0)")
        print("  /topk <int>           - Set top-k sampling")
        print("  /steps <int>          - Set inference steps (fewer = faster)")
        print("  /guided <prompt>      - Generate with prompt guidance")
        print("  /intermediate         - Show intermediate denoising steps")
        print("  /verbose              - Toggle verbose output")
        print("  /quit                 - Exit")
        print("-" * 60)
        
        verbose = True
        show_intermediate = False
        
        while True:
            try:
                user_input = input("\nCommand: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command_parts = user_input.split()
                    command = command_parts[0].lower()
                    
                    if command == '/quit':
                        print("Goodbye!")
                        break
                    elif command == '/verbose':
                        verbose = not verbose
                        print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                        continue
                    elif command == '/intermediate':
                        show_intermediate = not show_intermediate
                        print(f"Intermediate steps: {'ON' if show_intermediate else 'OFF'}")
                        continue
                    elif command == '/generate':
                        try:
                            length = int(command_parts[1]) if len(command_parts) > 1 else self.config.seq_len
                            print(f"\nGenerating {length} characters...")
                            print("-" * 40)
                            
                            result = self.generate(
                                seq_len=length,
                                verbose=verbose,
                                return_intermediate=show_intermediate
                            )
                            
                            print(f"\nGenerated text:")
                            print(f"'{result['text']}'")
                            
                            if verbose:
                                print(f"\nGeneration details:")
                                print(f"  Time: {result['generation_time']:.2f}s")
                                print(f"  Speed: {result['tokens_per_second']:.1f} tokens/s")
                                print(f"  Diffusion steps: {result['iterations']}")
                            
                            if show_intermediate and 'intermediate_texts' in result:
                                print(f"\nIntermediate denoising steps:")
                                for text in result['intermediate_texts']:
                                    print(f"  {text}")
                            
                        except (ValueError, IndexError):
                            print("Usage: /generate <length>")
                        continue
                    elif command == '/temp':
                        try:
                            temp = float(command_parts[1])
                            self.config.temperature = max(0.5, min(2.0, temp))
                            print(f"Temperature set to {self.config.temperature}")
                        except (ValueError, IndexError):
                            print("Usage: /temp <float>")
                        continue
                    elif command == '/topk':
                        try:
                            topk = int(command_parts[1])
                            self.config.top_k = max(1, min(100, topk))
                            print(f"Top-k set to {self.config.top_k}")
                        except (ValueError, IndexError):
                            print("Usage: /topk <int>")
                        continue
                    elif command == '/steps':
                        try:
                            steps = int(command_parts[1])
                            self.config.num_inference_steps = max(10, min(self.model.config.num_diffusion_steps, steps))
                            print(f"Inference steps set to {self.config.num_inference_steps}")
                        except (ValueError, IndexError):
                            print("Usage: /steps <int>")
                        continue
                    elif command == '/guided':
                        if len(command_parts) > 1:
                            prompt = ' '.join(command_parts[1:])
                            print(f"\nGenerating with guidance from: '{prompt}'")
                            print("-" * 40)
                            
                            guided_text = self.guided_generation(prompt, self.config.seq_len)
                            print(f"\nGuided generation:")
                            print(f"'{guided_text}'")
                        else:
                            print("Usage: /guided <prompt text>")
                        continue
                    else:
                        print("Unknown command. Type /quit to exit.")
                        continue
                else:
                    print("Please use commands starting with /. Try /generate to create text.")
                
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type /quit to exit or continue...")
                continue
            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def benchmark_speed(self, seq_lengths: list = [50, 100, 200], num_runs: int = 3):
        """Benchmark generation speed for different sequence lengths"""
        print(f"Benchmarking Diffusion LM speed with {num_runs} runs per length...")
        
        results = {}
        for seq_len in seq_lengths:
            print(f"\nTesting sequence length {seq_len}:")
            times = []
            
            for i in range(num_runs):
                result = self.generate(
                    seq_len=seq_len,
                    verbose=False
                )
                times.append(result['generation_time'])
                print(f"  Run {i+1}: {result['generation_time']:.2f}s ({result['tokens_per_second']:.1f} tokens/s)")
            
            avg_time = sum(times) / len(times)
            avg_speed = seq_len / avg_time
            
            results[seq_len] = {
                'avg_time': avg_time,
                'avg_speed': avg_speed,
                'times': times
            }
            
            print(f"  Average: {avg_time:.2f}s ({avg_speed:.1f} tokens/s)")
        
        return results

def main():
    """Main inference script"""
    config = DiffusionLMInferenceConfig(
        checkpoint_path="checkpoints_diffusion/diffusion_tiny_step_5000.pt",
        seq_len=150,
        num_inference_steps=100,  # Use fewer steps for faster inference
        compile_model=False
    )
    
    try:
        # Initialize inference
        inference = DiffusionLMInference(config)
        
        # Quick test generation
        print("\nTesting model with a quick generation...")
        test_result = inference.generate(seq_len=50, verbose=True)
        print(f"Test result: '{test_result['text'][:100]}...'")
        
        # Benchmark speed (will be slower than other models)
        print("\nRunning speed benchmark...")
        benchmark_results = inference.benchmark_speed(seq_lengths=[50, 100], num_runs=2)
        
        # Start interactive chat
        inference.chat_loop()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Trained Diffusion LM at the specified checkpoint path")
        print("2. Vocabulary file at: shakespeare_data/vocab.pkl")
        print("3. Run the Diffusion LM training script first to create the model")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# python -m inference.Diffusion_LM_chat