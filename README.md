# LLM Pretraining: GPT with Mixture of Experts  

This repository contains a clean, self-contained, and documented implementation of a Custom Generative Pre-trained Transformer (GPT) featuring a Sparse Mixture of Experts (MoE) routing layer. Developed initially for research and exploration within Google Colab, this architecture highlights how to implement Multi-Head Self Attention with Dense FeedForward OR Sparse Mixture-of-Experts network.  

***Why this matters***:
1) This allows anyone to have full control of their AI system, by having visibility over core model architecture and fixing any security vulnerabilities.
2) It has best performant LLM techniques (MHA, RoPE/YaRN and MoE), which can be used right away on a single GPU, to build custom small LLMs for specialized use cases, with Hallucination detection control.


## 🚀 Key Features  
* **Sparse MoE** Core: Custom deterministic token routing logic mapping tokens to top- $k$ expert sub-networks to increase model capacity without proportional compute costs.
* **Dropless Routing** for MoE: It processes every single token. If an expert is assigned 10x more tokens than others, the model calculates all of them. This ensures no information loss and makes it run faster in GPU based training.  
* **RoPE** and **YaRN** scaling: Advanced positional encoding to improve long-context coherence and relative positional tracking.  
* RMSNorm: Modern layer normalization applied globally for stabilizer scale-invariance during high-throughput training runs.  
* Hyperparameter Scaling Configuration: Integrated tuning layout designed to quickly experiment with model sizes, expert counts, and optimization boundaries.  


> *This new LLM variation is a combination of GPT-2, Grok and Mistral, where varies techniques were combined from those models.*  


*To-Do*: Add **Per-Layer QK-Norm** like Minimax M2.  



## Setup  

1. Install `uv`  

2. Clone repo and create `venv`  
    ```bash
    git clone https://github.com/miniscule-sys/llm-pretraining.git
    cd llm-pretraining 
    uv venv && uv init 
    ```  

3. Install deps:  
    ```
    uv sync
    ```  

Add any necessary lib through `uv add ...`  


## 🏗️ Architecture Design  

The MoE variant implements a Decoder-only Transformer framework with a sparse top- $k$ gating routing layer over multiple Feed-Forward Network (FFN) blocks, which uses *SwiGLU* activation.  
The standard GPT variant (*new*) has Dense FeedForward layers and option to switch between SwiGLU and GELU activations.  


```
[Token Input] -> [Embedding + RoPE] -> [Multi-Head Attention] -> [MoE Gate Router]
                                                                        |
                                                 --------------------------------
                                                |               |                |
                                            [Expert 1]      [Expert 2]   ... [Expert N]
                                                |               |                |
                                                 --------------------------------
                                                                |
                                                           [RMSNorm] -> [Token Prediction Output]
```  

## 🔑 Code Usage & Snippets  

### 1. Hyperparam selection and Training  

Provide directory paths to your data, tokenizer and model storage. 

```python
    class ProjectPaths:
        """Centralized configuration for absolute file paths."""
        base_dir: str = os.getcwd() #"./" # Change this to your root production directory 

        @property
        def tokenizer_dir(self) -> str:
            return os.path.join(self.base_dir, "custom_tokenizer")

        @property
        def model_checkpoints_dir(self) -> str:
            return os.path.join(self.base_dir, "trained_models")

        @property
        def data_corpus(self) -> str:
            return os.path.join(self.base_dir, "data", "corpus.txt")
```  

and your training loop args here:  
```python
    class TrainingArgs:
        """Strictly training-loop specific parameters."""
        tot_steps: int = 500
        batch_size: int = 2
        grad_accum_steps: int = 1
        lr_rate: float = 0.0005
        grad_clip: float = 1.0
```

Select whether MoE or Dense and provide model hyperparams:  
```python 
    # moe_args = MoEArgs(num_experts=2, top_k_experts=1, hidden_dim_multplier_perexp=2)
    moe_args = None # for Dense model training

    # Model architecture blueprint
    model_args = ModelArgs(
        vocab_size=safe_vocab_size,
        d_model=1024,
        num_heads=16,
        num_layers=8,
        hidden_dim_multiplier=6,
        org_context_length=1024,
        scale_factor=2.0,
        rotary_freq_base=10000,
        dropout=0.1,
        moe_args=moe_args,  # Comment this for Dense only
        device=device
    )

    # 3. Initialize Model & Optimizer
    model = MixGPTModel(model_args).to(device)
```  

The model integrates RoPE and YaRN scaling. So if `scale_factor` is 1, it uses RoPE, otherwise YaRN scaling.  
During inference, you can adjust `new_scale_factor` to get higher context length range.  

```python
    new_scale_factor:int = 0  # Adjust this if you want to extend context further during generation
    final_scale_factor = og_scale_factor + new_scale_factor
``` 

With `bf16` precision one can reach this max scale on a free Colab GPU:

```
org_context_length: 2048  
scale: 2  
model_dim: 1536  
num_heads: 16  
num_layers: 12  

(14.4 GB GPU)  
Total parameters: 459.86M
``` 

## 🧠 Inference  

Integrated `min_p` and `temp` filtering to select the best token.  

```python 
    prompt = "The cat (Felis"
    gen_token_count = 512

    print(f"\nGenerating with Min-P...")
    gen_st_time = time.perf_counter()

    token_ids = generate_minP(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt.strip(),
        device=device,
        max_new_tokens=gen_token_count,
        temp=0.5,
        min_p=0.1,
        use_cache=True
    )
```  

## 📈 Training Run resource usage comparison  


**MixGPT**  

For this config,  
```
org_context_length = 1024  
scale_factor = 2  
d_model = 1024  
num_heads = 16  
num_layers = 8  
hidden_dim_multiplier = 6

num_experts= 2 
top_k_experts = 1 
hidden_dim_multplier_perexp = 2

tot_steps = 2500  
batch_size = 2  
grad_accum_steps = 1  
```  

***MoE*** (Total param: 138.78M | Active: 88.45M)  
* Training runtime: 1458.85 s  
* GPU VRAM: 5.8 GB  
* Avg. Step time: 0.5725 s  

***Dense*** (Total param: 138.77M)  
* Training runtime: 1802.65 s  
* GPU VRAM: 8.2 GB  
* Avg. Step time: 0.6502 s


Fastest inference time was for Dense variant with kv-cache ON.  


________  

It will be under development with more small model implementation and experiemental results.  

