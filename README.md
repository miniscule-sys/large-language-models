# LLM Pretraining  

This work is an *ATTEMPT* to replicate SOTA LLMs. The source code is provided in a structured Jupyter Notebook, which can be converted into different modules easily.  

Developed initially for research and exploration within Google Colab, the collection contains source code of popular LLM techniques and their production implementation as a full-fledged LLM.  


### Small Language Models as the Future:  

1) This allows anyone to have full control of their AI system, by having visibility over core model architecture and fixing any security vulnerabilities.  
2) Includes widely adopted LLM techniques (MHA, MLA, RoPE/YaRN and MoE), which can be used right away on a **single GPU**, to build custom small LLMs for specialized use cases, with Hallucination detection control, giving control over ***token*** usage and independence over API reliance.  
3) Using scaling laws and Adapter modules, small language models can provide resource efficient and less halucination prone "Intelligence as a Service", which can be used in everyday compute environment, thus facilitating ***Environment Friendly***, low-cost and **fully-private** development.  


## 🚀 Key Features  
* Included both **Sparse MoE** and Dense Feed-Forward Network  
* **Dropless Routing** for MoE  
* **RoPE** positional embedding and **YaRN** scaling  
* **Training**, with cosine warm-up, gradient accumulation and weight initialization + **Autoregressive Decoding**, with *min_p* and *temp* scaling  
* __Architectures__: MLA, MHA, MoE, ... *more coming soon*  


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


________  

*Please reach out for collaboration and contribution*  

