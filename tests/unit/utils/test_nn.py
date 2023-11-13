from peft import PeftModelForCausalLM, PeftType
from transformers import LlamaConfig, LlamaForCausalLM

from src.xllm.core.config import Config
from src.xllm.utils.nn import apply_lora, stabilize_training


def test_apply_lora(llama_model_config: LlamaConfig):
    config = Config(apply_lora=True, raw_lora_target_modules="all")
    model = LlamaForCausalLM(config=llama_model_config)
    peft_model, lora_config = apply_lora(config=config, model=model)
    assert isinstance(peft_model, PeftModelForCausalLM)
    assert peft_model.peft_type == PeftType.LORA


def test_stabilize_training(llama_model_config: LlamaConfig):
    model = LlamaForCausalLM(config=llama_model_config)
    stabilize_training(model=model)
