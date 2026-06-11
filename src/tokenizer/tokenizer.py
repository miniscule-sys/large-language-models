""" Custom Tokenizer """

import os 
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast, AutoTokenizer 
import json
from typing import List, Iterator, Optional

class RustBPETokenizer:
    def __init__(self, vocab_size: int = 32000):
        # 1. Initialize BPE Model (Rust backend)
        # Using ByteLevel BPE is crucial for acting as a tiktoken replacement
        # as it handles UTF-8 bytes directly.
        self.tokenizer = Tokenizer(models.BPE())
        
        # 2. Pre-tokenization (ByteLevel)
        # Splits string into bytes, ensuring no <UNK> tokens for valid UTF-8
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # 3. Decoding
        self.tokenizer.decoder = decoders.ByteLevel()
        
        self.vocab_size = vocab_size
        self.chat_template = None

    def train(self, files: List[str], special_tokens: List[str] = None):
        """
        Train the tokenizer on a new text corpus (efficient Rust implementation).
        """
        if special_tokens is None:
            # Standard special tokens for pre/post training
            special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

        # Configure the Rust-based trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_tokens,
            # "min_frequency" can be tuned for corpus size
            min_frequency=2,
            show_progress=True
        )

        # Train directly from files (highly efficient streaming)
        self.tokenizer.train(files, trainer)
        print(f"Training complete. Vocab size: {self.tokenizer.get_vocab_size()}")

    def set_chat_template(self, template_str: str):
        """
        Assigns a Jinja2 template for post-training (chat/instruction tuning).
        """
        # Basic validation to ensure it's a valid Jinja string
        self.chat_template = template_str

    def save(self, directory: str):
        """
        Saves the tokenizer and the Jinja template.
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save the core tokenizer (vocab + merges)
        self.tokenizer.save(os.path.join(directory, "tokenizer.json"))
        
        # Save the config including the chat template and special tokens map
        # This mimics the transformers format for easy loading
        config = {
            "chat_template": self.chat_template,
            "model_type": "gpt2", # Architecture usually dictates this, generic here
            "vocab_size": self.tokenizer.get_vocab_size(),
            "special_tokens": [t.content for t in self.tokenizer.get_added_tokens_decoder().values()]
        }
        
        with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=4)
            
        print(f"Tokenizer and Chat Template saved to {directory}")

    def load_for_inference(self, directory: str):
        """
        Loads the tokenizer as a generic Hugging Face Fast Tokenizer.
        """
        # We use PreTrainedTokenizerFast to gain access to 'apply_chat_template'
        # and other high-level utility methods automatically.
        return PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(directory, "tokenizer.json"),
            tokenizer_config_file=os.path.join(directory, "tokenizer_config.json")
        )

# --- Usage Example --- 

# 1. Define a Jinja template (Llama-3 style example)
# This handles the "Post-training" requirement for chat formatting.
jinja_template = """
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
"""

# 2. Instantiate and Train
# Create a dummy file for demonstration
# with open("corpus.txt", "w") as f:
#     f.write("Hello world. This is a test corpus for training a new BPE tokenizer.")

tokenizer_wrapper = RustBPETokenizer(vocab_size=10000)
tokenizer_wrapper.train(["corpus.txt"])
tokenizer_wrapper.set_chat_template(jinja_template)

# # 3. Save
tokenizer_wrapper.save("./custom_tokenizer")

# 4. Inference 

# You can load it like any standard HF tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained("./my_custom_tokenizer_3")

# Test Chat Template (Post-training inference)
chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I am doing well!"}
]
formatted_chat = loaded_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
print("\n--- Chat Template Output ---") 
print(formatted_chat) 

# Test Raw Encoding (Pre-training/Efficiency check)
sample_text = "we might guess they relieved us humanely; but they think we are too dear" 

sample_text_2 = """
    SAMPLE 10: API Retry Logic

    User:
    Implement exponential backoff retry logic in Python.

    Assistant:
    ```python
    import random
    import time
"""

encoded = loaded_tokenizer.encode(sample_text) 
print(f"\n--- Encoded IDs ---\n{encoded}") 

decoded = loaded_tokenizer.decode(encoded)
print(f"\n--- Decoded Text ---\n{decoded}") 

