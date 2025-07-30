# Internal Recurrence & Annealing: An Exploration of Modern LLM Architectures

This repository is a hands-on exploration and implementation of several modern and experimental language model architectures. It goes beyond the standard Transformer to investigate alternative approaches to sequence modeling, including structured state spaces, iterative refinement, and diffusion-based generation.

The project is built in PyTorch and serves as a clear, educational, and self-contained codebase for understanding how these models work from the ground up. All models are trained on the "tiny-shakespeare" dataset.

## Models Implemented

This repository contains implementations for the following models:

1.  **nanoGPT (Baseline)**
    *   **Description:** A standard decoder-only Transformer model, based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). It serves as a powerful and well-understood baseline for comparison.
    *   **Core Concept:** Self-attention mechanism with quadratic complexity.

2.  **Iterative Thinking LLM**
    *   **Description:** An experimental model that refines its output over a series of "thinking" steps. Similar to Diffusion LMs, but with a common carryover between steps and an automatic stopping rule, making the temporal dimension to the internal process an implicit learnable process. 
    *   **Core Concept:** Internal recurrence and iterative refinement of hidden states before producing an output.

3.  **Diffusion-LM**
    *   **Description:** A language model based on diffusion principles, inspired by the [Diffusion-LM paper](https://arxiv.org/abs/2205.14217). It treats text generation as a denoising process, starting from random noise and gradually refining it into a coherent sequence of token embeddings.
    *   **Core Concept:** Denoising diffusion probabilistic models applied to discrete text data.

4.  **Mamba SSM**
    *   **Description:** An implementation of the Mamba architecture, a selective state-space model that has gained attention for its performance and linear-time complexity. It addresses the quadratic bottleneck of Transformers, allowing for much longer sequence lengths.
    *   **Core Concept:** Selective State Space Model (SSM) that allows for content-aware reasoning. Based on the paper [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).

## Project Structure

```
Internal Recurrence & Annealing/
├── models/
│   ├── nanoGPT.py
│   ├── Iterative_Thinking_LLM.py
│   ├── Diffusion_LM.py
│   └── MAMBA_SSM.py
├── training/
│   ├── nanoGPT_trainer.py
│   ├── iterative_thinking_trainer.py
│   ├── Diffusion_LM_trainer.py
│   └── MAMBA_SSM_trainer.py
├── inference/
│   ├── nanoGPT_chat.py
│   ├── iterative_thinking_chat.py
│   └── Diffusion_LM_chat.py
├── data/
│   ├── data_prep.py
│   └── data_class.py
└── README.md
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ScottBiggs2/Building-LLMs.git
    cd "Internal Recurrence & Annealing"
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    *It is recommended to create a `requirements.txt` file with necessary packages like `torch`, `numpy`, `tqdm`, `wandb`, etc.*
    ```bash
    pip install torch numpy tqdm wandb einops
    ```

## How to Use

### 1. Prepare the Data

First, run the data preparation script. This will download the "tiny-shakespeare" dataset and create tokenized `train_encoded.npy`, `val_encoded.npy`, and `vocab.pkl` files inside the `shakespeare_data/` directory.

```bash
python -m data.data_prep
```

### 2. Train a Model

You can train any of the implemented models using their respective trainer scripts. Training configurations (like model size, batch size, learning rate) can be modified directly within each trainer's `main()` function.

**Train nanoGPT:**
```bash
python -m training.nanoGPT_trainer
```

**Train Iterative Thinking LLM:**
```bash
python -m training.iterative_thinking_trainer
```

**Train Diffusion-LM:**
```bash
python -m training.Diffusion_LM_trainer
```

**Train Mamba:**
*Note: For a significant speedup on Mamba, ensure `compile_model=True` is set in the trainer script (requires PyTorch 2.0+).*
```bash
python -m training.MAMBA_SSM_trainer
```

Checkpoints for each model will be saved in their corresponding `checkpoints_<model_name>/` directory.

### 3. Chat with a Trained Model

Once a model is trained, you can interact with it using the chat scripts. These scripts will automatically load the best-performing checkpoint.

**Chat with nanoGPT:**
```bash
python -m inference.nanoGPT_chat
```

**Chat with Iterative Thinking LLM:**
```bash
python -m inference.iterative_thinking_chat
```

**Chat with Diffusion-LM:**
```bash
python -m inference.Diffusion_LM_chat
```

## Future Work

-   [ ] **Mamba Chat Interface:** Create a chat script (`MAMBA_SSM_chat.py`) to interact with the trained Mamba model.
-   [ ] **Parallel Scan for Mamba:** Implement a true parallel scan algorithm to replace the sequential fallback for maximum training efficiency.
-   [ ] **Formal Benchmarking:** Conduct a more rigorous comparison of the models on perplexity, generation quality, and inference speed.
-   [ ] **Requirements File:** Add a `requirements.txt` for easier dependency management.

---

This project is for educational and research purposes. Contributions and feedback are welcome!