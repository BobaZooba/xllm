import os

from dotenv import load_dotenv

load_dotenv()


PROJECT_DIR: str = os.getcwd()

STATIC_DIR: str = os.path.join(PROJECT_DIR, "tests/static/")
TOKENIZERS_DIR: str = os.path.join(STATIC_DIR, "tokenizers/")

LLAMA_TOKENIZER_DIR: str = os.path.join(TOKENIZERS_DIR, "llama/")
FALCON_TOKENIZER_DIR: str = os.path.join(TOKENIZERS_DIR, "falcon/")

LORA_FOR_LLAMA_DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
