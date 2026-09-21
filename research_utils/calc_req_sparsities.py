"""Parameter budgets for residual-stream slicing (no checkpoint downloads).

Qwen3.5 presets describe the dense text models used by TrueSlice, not the vision
encoder or MTP training weights. Linear attention here means Qwen3.5 Gated DeltaNet.
Hybrid SliceGPT counts are analytical estimates of its shortcut layout, not a
claim that the repository's SliceGPT adapters support those models.
"""

import math

models = {
    # "Phi-2 (2.7B)": {
    #     "vocab_size": 51200, "hidden_dim": 2560, "intermediate_dim": 10240,
    #     "num_layers": 32, "num_q_heads": 32, "num_kv_heads": 32, 
    #     "mlp_type": "standard", "is_parallel": True,
    #     "q_bias": True, "k_bias": True, "v_bias": True, "dense_bias": True, 
    #     "fc1_bias": True, "fc2_bias": True, "lm_head_bias": True, "norm_type": "layernorm"
    # },
    # "OPT (6.7B)": {
    #     "vocab_size": 50272, "hidden_dim": 4096, "intermediate_dim": 16384,
    #     "num_layers": 32, "num_q_heads": 32, "num_kv_heads": 32, 
    #     "mlp_type": "standard", "q_bias": True, "k_bias": True, "v_bias": True,
    #     "dense_bias": True, "fc1_bias": True, "fc2_bias": True,
    #     "lm_head_bias": True, "pos_embeddings": 2050, "norm_type": "layernorm",
    # },
    # "Phi-3-Mini (3.8B)": {
    #     "vocab_size": 32064, "hidden_dim": 3072, "intermediate_dim": 8192,
    #     "num_layers": 32, "num_q_heads": 32, "num_kv_heads": 32, 
    #     "mlp_type": "gated",
    # },
    # "Phi-3-Small (7B)": {
    #     "vocab_size": 100352, "hidden_dim": 4096, "intermediate_dim": 14336,
    #     "num_layers": 32, "num_q_heads": 32, "num_kv_heads": 8, 
    #     "mlp_type": "gated",
    # },
    "Llama-3.2 (1B)": {
        "vocab_size": 128256, "hidden_dim": 2048, "intermediate_dim": 8192,
        "num_layers": 16, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64,
        "mlp_type": "gated", "tied": True
    },
    "Llama-3.2 (3B)": {
        "vocab_size": 128256, "hidden_dim": 3072, "intermediate_dim": 8192,
        "num_layers": 28, "num_q_heads": 24, "num_kv_heads": 8, "head_dim": 128,
        "mlp_type": "gated", "tied": True
    },
    "Llama-3.1 (8B)": {
        "vocab_size": 128256, "hidden_dim": 4096, "intermediate_dim": 14336,
        "num_layers": 32, "num_q_heads": 32, "num_kv_heads": 8, 
        "mlp_type": "gated",
    },
    "Llama-3.1 (70B)": {
        "vocab_size": 128256, "hidden_dim": 8192, "intermediate_dim": 28672,
        "num_layers": 80, "num_q_heads": 64, "num_kv_heads": 8, 
        "mlp_type": "gated",
    },
    "qwen3-1.7b": {
        "vocab_size": 151936, "hidden_dim": 2048, "intermediate_dim": 6144,
        "num_layers": 28, "num_q_heads": 16, "num_kv_heads": 8, "head_dim": 128,
        "mlp_type": "gated", "tied": True, "norm_type": "rmsnorm", "qk_norms": True,
    },
    "qwen3-4b": {
        "vocab_size": 151936, "hidden_dim": 2560, "intermediate_dim": 9728,
        "num_layers": 36, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "mlp_type": "gated", "tied": True, "norm_type": "rmsnorm", "qk_norms": True,
    },
    "qwen3-8b": {
        "vocab_size": 151936, "hidden_dim": 4096, "intermediate_dim": 12288,
        "num_layers": 36, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "mlp_type": "gated", "tied": False, "norm_type": "rmsnorm", "qk_norms": True,
    },
    "qwen3-14b": {
        "vocab_size": 151936, "hidden_dim": 5120, "intermediate_dim": 17408,
        "num_layers": 40, "num_q_heads": 40, "num_kv_heads": 8, "head_dim": 128,
        "mlp_type": "gated", "tied": False, "norm_type": "rmsnorm", "qk_norms": True,
    },
    "qwen3-32b": {
        "vocab_size": 151936, "hidden_dim": 5120, "intermediate_dim": 25600,
        "num_layers": 64, "num_q_heads": 40, "num_kv_heads": 8, "head_dim": 128,
        "mlp_type": "gated", "tied": False, "norm_type": "rmsnorm", "qk_norms": True,
    },

    # Official text configs, checked 2026-09-21:
    # https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json
    "qwen3.5-4B": {
        "vocab_size": 248320, "hidden_dim": 2560, "intermediate_dim": 9216,
        "num_layers": 32, "num_q_heads": 16, "num_kv_heads": 4, "head_dim": 256,
        "mlp_type": "gated", "tied": True, "norm_type": "rmsnorm", "qk_norms": True,
        "attn_output_gate": True,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    },
    # https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json
    "qwen3.5-9B": {
        "vocab_size": 248320, "hidden_dim": 4096, "intermediate_dim": 12288,
        "num_layers": 32, "num_q_heads": 16, "num_kv_heads": 4, "head_dim": 256,
        "mlp_type": "gated", "tied": False, "norm_type": "rmsnorm", "qk_norms": True,
        "attn_output_gate": True,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    },
}


def calc_attn_params(in_d, out_d, config, d_q, d_kv):
    # Qwen3.5 full attention projects both queries and an output gate.
    q_width = d_q * (2 if config.get("attn_output_gate", False) else 1)
    params = (q_width + 2 * d_kv) * in_d
    params += out_d * d_q
    if config.get("q_bias", False): params += q_width
    if config.get("k_bias", False): params += d_kv
    if config.get("v_bias", False): params += d_kv
    if config.get("dense_bias", False): params += out_d
    return params


def get_layer_types(config):
    """Explicit hybrid layout, or homogeneous attention for older presets."""
    layer_types = config.get("layer_types")
    if layer_types is None:
        layer_types = [config.get("attention_type", "full_attention")] * config["num_layers"]
    if len(layer_types) != config["num_layers"] or not layer_types:
        raise ValueError("layer_types must contain one entry per decoder layer")
    supported = {"full_attention", "sliding_attention", "linear_attention"}
    if any(kind not in supported for kind in layer_types):
        raise ValueError(f"Unsupported attention layer types: {layer_types}")
    return layer_types


def get_attention_dimensions(config, d_model):
    """Original full-attention widths; pure DeltaNet models need no Q/K/V head fields."""
    if all(kind == "linear_attention" for kind in get_layer_types(config)):
        return 0, 0
    head_dim = config.get("head_dim", d_model // config["num_q_heads"])
    return config["num_q_heads"] * head_dim, config["num_kv_heads"] * head_dim


def calc_linear_attn_params(in_d, out_d, config):
    """Gated DeltaNet projections, depthwise convolution and learned state parameters.

    Internal widths stay fixed when slicing the residual stream. Recurrent cache
    tensors are runtime state, not model parameters. The gated norm is counted
    separately in get_attn_norm_params().
    """
    key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    value_dim = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    value_heads = config["linear_num_value_heads"]
    qkv_dim = 2 * key_dim + value_dim
    # in_proj_qkv, in_proj_z, in_proj_b, in_proj_a; then out_proj.
    params = (qkv_dim + value_dim + 2 * value_heads) * in_d + value_dim * out_d
    params += qkv_dim * config["linear_conv_kernel_dim"]
    params += 2 * value_heads  # dt_bias and A_log
    return params


def calc_mixer_params(in_d, out_d, config, d_q, d_kv, layer_type):
    if layer_type == "linear_attention":
        return calc_linear_attn_params(in_d, out_d, config)
    return calc_attn_params(in_d, out_d, config, d_q, d_kv)


def calc_mlp_params(in_d, out_d, config, I):
    params = 0
    mult = 2 if config.get("mlp_type") == "gated" else 1
    params += mult * I * in_d
    if config.get("fc1_bias", False): params += mult * I
    params += out_d * I
    if config.get("fc2_bias", False): params += out_d
    return params


def get_norm_params(config, d_new):
    L = config["num_layers"]

    params_per_norm = d_new
    if config.get("norm_type", "rmsnorm") == "layernorm":
        params_per_norm = params_per_norm * 2

    if config.get("is_parallel", False):
        norm_params = (L * params_per_norm ) + params_per_norm
    else:
        norm_params = (2 * L * params_per_norm) + params_per_norm

    return norm_params


def get_attn_norm_params(config, d_model):
    """Internal norms use original head dimensions, regardless of residual slicing."""
    norm_params = 0
    for layer_type in get_layer_types(config):
        if layer_type == "linear_attention":
            # DeltaNet's gated RMSNorm is shared over value heads.
            norm_params += config["linear_value_head_dim"]
        elif config.get("qk_norms", False):
            head_dim = config.get("head_dim", d_model // config["num_q_heads"])
            norm_params += 2 * head_dim * (2 if config.get("norm_type", "rmsnorm") == "layernorm" else 1)
    return norm_params


def calc_global_params(config, d_model) -> int:
    L = config["num_layers"]
    I = config["intermediate_dim"]
    V = config["vocab_size"]

    d_q, d_kv = get_attention_dimensions(config, d_model)

    # Global embeddings and head params may be tied
    pos_embs = config.get("pos_embeddings", 0)
    embedding_params = (V * d_model) + (pos_embs * d_model)

    global_head_params = 0
    if not config.get("tied", False):
        global_head_params = V * d_model

    if config.get("lm_head_bias", False):
        global_head_params += V

    # Global attn and mlp param
    global_attn_params = sum(
        calc_mixer_params(d_model, d_model, config, d_q, d_kv, kind) for kind in get_layer_types(config)
    )
    global_mlp_params = calc_mlp_params(d_model, d_model, config, I) * L

    # Global norm params
    global_norm_params = get_norm_params(config, d_model) + get_attn_norm_params(config, d_model)

    # Hopefully not missing any other static parameters
    return embedding_params + global_head_params + global_attn_params + global_mlp_params + global_norm_params


def calc_trueslice_params(config, d_model, d_new) -> int:
    L = config["num_layers"]
    I = config["intermediate_dim"]
    V = config["vocab_size"]
    
    d_q, d_kv = get_attention_dimensions(config, d_model)

    # Trueslice embeddings and head params are forceably untied during process
    pos_embs = config.get("pos_embeddings", 0)
    trueslice_embedding_params = (V * d_new) + (pos_embs * d_new)

    trueslice_head_params = V * d_new

    if config.get("lm_head_bias", False):
        trueslice_head_params += V

    # Trueslice attn and mlp param
    trueslice_attn_params = sum(
        calc_mixer_params(d_new, d_new, config, d_q, d_kv, kind) for kind in get_layer_types(config)
    )
    trueslice_mlp_params = calc_mlp_params(d_new, d_new, config, I) * L

    # Trueslice norm params keeps the norms inside the attn space untouched
    trueslice_norm_params = get_norm_params(config, d_new) + get_attn_norm_params(config, d_model)

    # Hopefully not missing any other static parameters
    return trueslice_embedding_params + trueslice_head_params + trueslice_attn_params + trueslice_mlp_params + trueslice_norm_params


def calc_slicegpt_params(config, d_model, d_new):
    L = config["num_layers"]
    I = config["intermediate_dim"]
    V = config["vocab_size"]
    
    d_q, d_kv = get_attention_dimensions(config, d_model)

    is_parallel = config.get("is_parallel", False)

    # Slicegpt embeddings and head params are forceably untied during process
    pos_embs = config.get("pos_embeddings", 0)
    embedding_params = (V * d_new) + (pos_embs * d_new)

    slicegpt_head_params = V * d_model

    if config.get("lm_head_bias", False):
        slicegpt_head_params += V

    # Slicegpt attn and mlp params of first L - 1 layers
    layer_types = get_layer_types(config)
    standard_mlp = calc_mlp_params(d_new, d_new, config, I)
    slicegpt_layer_params = sum(
        calc_mixer_params(d_new, d_new, config, d_q, d_kv, kind) + standard_mlp
        for kind in layer_types[:-1]
    )

    # Slicegpt attn and mlp params of final layer
    last_attn_out_d = d_model if is_parallel else d_new
    slicegpt_layer_params += calc_mixer_params(d_new, last_attn_out_d, config, d_q, d_kv, layer_types[-1])
    slicegpt_layer_params += calc_mlp_params(d_new, d_model, config, I)

    # Slicegpt shortcut params of first L - 1 layers
    shortcut_Q = d_new ** 2
    slicegpt_shortcut_params = shortcut_Q * (L - 1) if is_parallel else (2 * shortcut_Q) * (L - 1)

    # Slicegpt shortcut params of final layer
    slicegpt_shortcut_params += (d_new * d_model)
    if not is_parallel:
        slicegpt_shortcut_params += (d_new ** 2)

    # Slicegpt norm params keeps the norms inside the attn space untouched
    slicegpt_norm_params = get_attn_norm_params(config, d_model)

    # Hopefully not missing any other static parameters
    return embedding_params + slicegpt_head_params + slicegpt_layer_params + slicegpt_shortcut_params + slicegpt_norm_params


def find_optimal_sparsities(config, global_target_sparsity, dense_param_count, block_size=1):
    """
    Uses binary search to find the closest dimension that yields a parameter 
    count >= the global target, drastically reducing calculation overhead.
    """
    d_model = config["hidden_dim"]
    if not isinstance(block_size, int) or not 1 <= block_size <= d_model:
        raise ValueError("block_size must be an integer between 1 and hidden_dim")
    if not 0 <= global_target_sparsity <= 1:
        raise ValueError("global_target_sparsity must be between 0 and 1")
    if dense_param_count <= 0:
        raise ValueError("dense_param_count must be positive")

    global_target_params = math.ceil(dense_param_count * (1 - global_target_sparsity) / block_size) * block_size

    # Find best trueslice cut, param count must be at minimum equal to global target param count
    best_d_trueslice = d_model
    best_params_trueslice = None
    
    low_blocks = 1
    high_blocks = d_model // block_size
    
    while low_blocks <= high_blocks:
        mid_blocks = (low_blocks + high_blocks) // 2
        d_test = mid_blocks * block_size
        
        trueslice_params = calc_trueslice_params(config, d_model, d_test)
        
        if trueslice_params >= global_target_params:
            best_d_trueslice = d_test
            best_params_trueslice = trueslice_params
            high_blocks = mid_blocks - 1 
        else:
            low_blocks = mid_blocks + 1

    if best_params_trueslice is None:
        raise ValueError("No aligned TrueSlice width reaches the requested parameter budget")

    raw_sparsity_trueslice = 1 - (best_d_trueslice / d_model)
    sparsity_trueslice = math.ceil(raw_sparsity_trueslice * 1000000) / 1000000

    # Find best slicegpt cut, param count must be at minimum equal to trueslice param count
    best_d_slicegpt = d_model
    best_params_slicegpt = None
    
    low_blocks = 1
    high_blocks = d_model // block_size
    
    while low_blocks <= high_blocks:
        mid_blocks = (low_blocks + high_blocks) // 2
        d_test = mid_blocks * block_size
        
        slicegpt_params = calc_slicegpt_params(config, d_model, d_test)
        
        if slicegpt_params >= best_params_trueslice:
            best_d_slicegpt = d_test
            best_params_slicegpt = slicegpt_params
            high_blocks = mid_blocks - 1 
        else:
            low_blocks = mid_blocks + 1
            
    if best_params_slicegpt is None:
        raise ValueError("No aligned SliceGPT width reaches the matched TrueSlice parameter budget")

    raw_sparsity_slicegpt = 1 - (best_d_slicegpt / d_model)
    sparsity_slicegpt = math.ceil(raw_sparsity_slicegpt * 1000000) / 1000000
    
    return {
        "global_target_params": global_target_params,
        "best_d_trueslice": best_d_trueslice,
        "best_params_trueslice": best_params_trueslice,
        "required_sparsity_trueslice": sparsity_trueslice,
        "actual_trueslice_cut": 1 - (best_params_trueslice / dense_param_count),
        "best_d_slicegpt": best_d_slicegpt,
        "best_params_slicegpt": best_params_slicegpt,
        "required_sparsity_slicegpt": sparsity_slicegpt,
        "actual_slicegpt_cut": 1 - (best_params_slicegpt / dense_param_count)
    }


def main():
    sparsities = [0.1, 0.2, 0.3]
    # sparsities = [0.125, 0.25, 0.375]
    block_size = 1

    for name, cfg in models.items():
        # Calculate the base dense parameter count once
        dense_param_count = calc_global_params(cfg, cfg["hidden_dim"])
    
        print(f"\n{name} (Base: {dense_param_count:,})")
        if "linear_attention" in get_layer_types(cfg):
            print("Dense text model only; hybrid SliceGPT columns are analytical estimates.")

        list_trueslice = []
        list_slicegpt = []
    
        # Define and print the headers
        headers = (
            f"{'Target %':<9} | "
            f"{'Targ Params':<14} | "
            f"{'TS D':<6} | "
            f"{'TS Params':<14} | "
            f"{'Req TS':<10} | "
            f"{'Act TS':<10} | "
            f"{'SG D':<6} | "
            f"{'SG Params':<14} | "
            f"{'Req SG %':<10} | "
            f"{'Act SG %'}"
        )
        print(headers)
        print("-" * len(headers))
    
        for s in sparsities:
            result = find_optimal_sparsities(cfg, s, dense_param_count, block_size=block_size)

            list_trueslice.append(f"{result['required_sparsity_trueslice']:>9.6f}")
            list_slicegpt.append(f"{result['required_sparsity_slicegpt']:>9.6f}")
        
            print(f"{s * 100:>5.1f}%    | "
                  f"{result['global_target_params']:<14,} | "
                  f"{result['best_d_trueslice']:<6} | "
                  f"{result['best_params_trueslice']:<14,} | "
                  f"{result['required_sparsity_trueslice']:>9.6f} | "
                  f"{result['actual_trueslice_cut'] * 100:>9.4f}\\% | "
                  f"{result['best_d_slicegpt']:<6} | "
                  f"{result['best_params_slicegpt']:<14,} | "
                  f"{result['required_sparsity_slicegpt']:>9.6f} | "
                  f"{result['actual_slicegpt_cut'] * 100:>8.4f}\\%")

        trueslice_str = " ".join(x.strip() for x in list_trueslice)
        slicegpt_str = " ".join(x.strip() for x in list_slicegpt)

        print(f"SPARSITIES_TRUESLICE=\"{trueslice_str}\"")
        print(f"SPARSITIES_SLICEGPT=\"{slicegpt_str}\"")

if __name__ == "__main__":
    main()
