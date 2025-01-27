import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from k_model import Transformer, ModelConfig

warnings.filterwarnings('ignore', category=UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_transformers_model():
    ModelConfig.register_for_auto_class()
    Transformer.register_for_auto_class("AutoModelForCausalLM")

    lm_config = ModelConfig()
    lm_model = Transformer(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = 'sft_model/sft_dim768_layers12_vocab_size6144.pth'

    state_dict = torch.load(ckpt_path, map_location=device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    lm_model.load_state_dict(state_dict, strict=False)
    print(f'模型参数: {count_parameters(lm_model) / 1e6} 百万 = {count_parameters(lm_model) / 1e9} B (Billion)')

    lm_model.save_pretrained("Tiny-K", safe_serialization=False)


def export_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k', trust_remote_code=True, use_fast=False)
    tokenizer.save_pretrained("Tiny-K")


if __name__ == '__main__':
    # 1
    export_transformers_model()
    # 2
    export_tokenizer()
    # # 3
    # push_to_hf()
