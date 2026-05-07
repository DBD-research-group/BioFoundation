import torch
from safetensors.torch import load_file

def convert(safetensors_path: str, pt_path: str):
    print(f"Loading weights from {safetensors_path}...")
    
    # Load the raw safetensors file into a state dictionary
    raw_state_dict = load_file(safetensors_path)
    
    # 1. Filter and strip the "beats." prefix to isolate the audio encoder
    filtered_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("beats."):
            # Remove the "beats." prefix
            new_key = key[6:]
            filtered_state_dict[new_key] = value
            
    print(f"Extracted {len(filtered_state_dict)} BEATs parameters from {len(raw_state_dict)} total parameters.")
    
    # 2. BEATs configuration extracted from HF: https://huggingface.co/EarthSpeciesProject/NatureLM-audio/blob/main/config.json
    beats_cfg = {
        "activation_dropout": 0.0,
        "activation_fn": "gelu",
        "attention_dropout": 0.0,
        "conv_bias": False,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "deep_norm": True,
        "dropout": 0.0,
        "dropout_input": 0.0,
        "embed_dim": 512,
        "encoder_attention_heads": 12,
        "encoder_embed_dim": 768,
        "encoder_ffn_embed_dim": 3072,
        "encoder_layerdrop": 0.05,
        "encoder_layers": 12,
        "finetuned_model": True,
        "gru_rel_pos": True,
        "input_patch_size": 16,
        "layer_norm_first": False,
        "layer_wise_gradient_decay_ratio": 0.6,
        "max_distance": 800,
        "num_buckets": 320,
        "predictor_class": 527,
        "predictor_dropout": 0.0,
        "relative_position_embedding": True
    }
    
    # 3. Package the filtered weights and the config into a single dictionary
    checkpoint = {
        "model": filtered_state_dict,
        "cfg": beats_cfg
    }
    
    print(f"Saving to {pt_path}...")
    torch.save(checkpoint, pt_path)

# File paths
input_model = "/workspace/models/beats/model.safetensors"
output_model = "/workspace/models/beats/beats_naturelm.pt"

if __name__ == "__main__":
    convert(input_model, output_model)