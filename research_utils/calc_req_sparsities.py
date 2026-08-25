import math

models = {
    "TinyLlama": {
        "vocab_size": 32000, "hidden_dim": 2048, "intermediate_dim": 5632,
        "num_layers": 22, "num_q_heads": 32, "num_kv_heads": 4, 
        "mlp_type": "gated", "target_base": 1099956224
    },
    "Llama-3.2 (1B)": {
        "vocab_size": 128256, "hidden_dim": 2048, "intermediate_dim": 8192,
        "num_layers": 16, "num_q_heads": 32, "num_kv_heads": 8, 
        "mlp_type": "gated", "target_base": 1498415104
    },
    "Phi-2 (2.7B)": {
        "vocab_size": 51200, "hidden_dim": 2560, "intermediate_dim": 10240,
        "num_layers": 32, "num_q_heads": 32, "num_kv_heads": 32, 
        "mlp_type": "standard", "is_parallel": True, "target_base": 2779514880,
        "q_bias": True, "k_bias": True, "v_bias": True, "dense_bias": True, 
        "fc1_bias": True, "fc2_bias": True, "lm_head_bias": True
    },
    "Llama-3.2 (3B)": {
        "vocab_size": 128256, "hidden_dim": 3072, "intermediate_dim": 8192,
        "num_layers": 28, "num_q_heads": 24, "num_kv_heads": 8, 
        "mlp_type": "gated", "target_base": 3606577152
    },
    "Phi-3-Mini (3.8B)": {
        "vocab_size": 32064, "hidden_dim": 3072, "intermediate_dim": 8192,
        "num_layers": 32, "num_q_heads": 32, "num_kv_heads": 32, 
        "mlp_type": "gated", "target_base": 3820879872
    },
    "OPT (6.7B)": {
        "vocab_size": 50272, "hidden_dim": 4096, "intermediate_dim": 16384,
        "num_layers": 32, "num_q_heads": 32, "num_kv_heads": 32, 
        "mlp_type": "standard", "target_base": 6863905888,
        "q_bias": True, "k_bias": True, "v_bias": True, "dense_bias": True, 
        "fc1_bias": True, "fc2_bias": True, "lm_head_bias": True, "pos_embeddings": 2050
    },
    "Llama-3.2 (8B)": {
        "vocab_size": 128256, "hidden_dim": 4096, "intermediate_dim": 14336,
        "num_layers": 32, "num_q_heads": 32, "num_kv_heads": 8, 
        "mlp_type": "gated", "target_base": 8029995008
    }
}

def calc_attn_params(in_d, out_d, config, d_q, d_kv):
    params = (d_q + 2 * d_kv) * in_d
    params += out_d * d_q
    if config.get("q_bias", False): params += d_q
    if config.get("k_bias", False): params += d_kv
    if config.get("v_bias", False): params += d_kv
    if config.get("dense_bias", False): params += out_d
    return params

def calc_mlp_params(in_d, out_d, config, I):
    params = 0
    mult = 2 if config.get("mlp_type") == "gated" else 1
    params += mult * I * in_d
    if config.get("fc1_bias", False): params += mult * I
    params += out_d * I
    if config.get("fc2_bias", False): params += out_d
    return params

def get_matrix_params_for_d(config, d_new):
    """Calculates exactly how many pure matrix parameters exist at dimension d_new."""
    d_model = config["hidden_dim"]
    L = config["num_layers"]
    I = config["intermediate_dim"]
    V = config["vocab_size"]
    
    d_q = d_model
    d_kv = config["num_kv_heads"] * (d_model // config["num_q_heads"])
    is_parallel = config.get("is_parallel", False)

    # 1. Embeddings & Head
    pos_embs = config.get("pos_embeddings", 0)
    embedding_params = (V * d_new) + (pos_embs * d_new)

    slicegpt_head_params = V * d_model
    global_head_params = V * d_new

    if config.get("lm_head_bias", False):
        slicegpt_head_params += V
        global_head_params += V

    standard_attn = calc_attn_params(d_new, d_new, config, d_q, d_kv)
    standard_mlp = calc_mlp_params(d_new, d_new, config, I)
    
    global_layer_params = (standard_attn + standard_mlp) * L
    slicegpt_layer_params = (standard_attn + standard_mlp) * (L - 1)
    
    shortcut_Q = d_new ** 2
    slicegpt_layer_params += shortcut_Q * (L - 1) if is_parallel else (2 * shortcut_Q) * (L - 1)

    last_attn_out_d = d_model if is_parallel else d_new
    slicegpt_layer_params += calc_attn_params(d_new, last_attn_out_d, config, d_q, d_kv)
    slicegpt_layer_params += calc_mlp_params(d_new, d_model, config, I)
    
    slicegpt_layer_params += (d_new * d_model)
    if not is_parallel:
        slicegpt_layer_params += (d_new ** 2)

    return (embedding_params + slicegpt_head_params + slicegpt_layer_params, 
            embedding_params + global_head_params + global_layer_params)


def find_optimal_slicegpt_sparsity(model_name, config, target_global_sparsity, block_size=1):
    """
    Uses binary search to find the closest dimension that yields a parameter 
    count >= the global target, drastically reducing calculation overhead.
    """
    d_model = config["hidden_dim"]
    
    _, dense_global = get_matrix_params_for_d(config, d_model)
    static_params = config["target_base"] - dense_global
    
    d_target = math.ceil(d_model * (1 - target_global_sparsity) / block_size) * block_size
    _, global_target_matrices = get_matrix_params_for_d(config, d_target)
    global_target_params = global_target_matrices + static_params
    
    best_d = d_model
    best_slice_params = config["target_base"]
    
    low_blocks = 1
    high_blocks = d_model // block_size
    
    while low_blocks <= high_blocks:
        mid_blocks = (low_blocks + high_blocks) // 2
        d_test = mid_blocks * block_size
        
        slice_matrices, _ = get_matrix_params_for_d(config, d_test)
        slice_params = slice_matrices + static_params
        
        if slice_params >= global_target_params:
            best_d = d_test
            best_slice_params = slice_params
            high_blocks = mid_blocks - 1 
        else:
            low_blocks = mid_blocks + 1
            
    required_sparsity = 1 - (best_d / d_model)
    
    return {
        "global_target": global_target_params,
        "slice_d": best_d,
        "slice_params": best_slice_params,
        "slice_sparsity": required_sparsity
    }


sparsities = [0.125, 0.25, 0.375, 0.50]
block_size = 1

for name, cfg in models.items():
    print(f"\n{name} (Base: {cfg['target_base']:,})")
    print(f"{'Global Target':<15} | {'Target Params':<15} | {'Req. SliceGPT %':<17} | {'New Dim':<10} | {'SliceGPT Params':<17} | {'Diff'}")
    print("-" * 94)
    
    for s in sparsities:
        result = find_optimal_slicegpt_sparsity(name, cfg, s, block_size=block_size)
        
        diff = result['slice_params'] - result['global_target']
        str_diff = f"+{diff:,}" if diff >= 0 else f"{diff:,}"
        
        print(f"{s * 100:>6.1f}% Target | "
              f"{result['global_target']:<15,} | "
              f"{result['slice_sparsity'] * 100:>6.4f}%           | "
              f"{result['slice_d']:<10} | "
              f"{result['slice_params']:<17,} | "
              f"{str_diff}")