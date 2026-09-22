"""Weight export for the paper\'s Qwen3.5 SFT student (not an inference model registry)."""
from .processors import quantize_params, remove_padding
from .qwen3_5 import convert_qwen3_5_to_hf


def postprocess_hf_param(args, megatron_param_name, hf_param_name, param):
    return remove_padding(megatron_param_name, param, args.vocab_size)


def convert_to_hf(args, model_name, name, param, quantization_config=None):
    if "qwen3_5" not in model_name:
        raise ValueError(f"The bundled training exporter supports Qwen3.5 only, not {model_name}")
    param = remove_padding(name, param, args.vocab_size)
    converted = convert_qwen3_5_to_hf(args, name, param)
    return quantize_params(args, name, converted, quantization_config)
