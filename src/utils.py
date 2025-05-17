from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple, Optional
import os
from cnn_gemma import CNNGemmaForConditionalGeneration, CNNGemmaConfig
from huggingface_hub import snapshot_download
import torch

def download_model(model_path: str, repo_id: str):

    snapshot_download(
        repo_id=repo_id,
        local_dir=model_path,
        ignore_patterns="*.gitattributes"
    )

    return

def load_hf_model(model_path: str, repo_id: str, device: str, dtype: Optional[torch.dtype] = torch.bfloat16) -> Tuple[CNNGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", padding_side="right")
    assert tokenizer.padding_side == "right"

    os.makedirs(model_path, exist_ok=True)

    #download safetensors and config.json if not downloaded
    if not any(
        f.endswith('.safetensors') for f in os.listdir(model_path)
        if os.path.isfile(os.path.join(model_path, f))
    ):
        download_model(model_path, repo_id)

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = CNNGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = CNNGemmaForConditionalGeneration(config).to(dtype).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)


def load_pretrained_model(paligemma_path: str, config: CNNGemmaConfig, device: str, dtype: Optional[torch.dtype] = torch.bfloat16) -> Tuple[CNNGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", padding_side="right")
    assert tokenizer.padding_side == "right"

    os.makedirs(paligemma_path, exist_ok=True)

    #download paligemma safetensors and config.json if not downloaded
    if not any(
        f.endswith('.safetensors') for f in os.listdir(paligemma_path)
        if os.path.isfile(os.path.join(paligemma_path, f))
    ):
        revision = "bfloat16"
        if (dtype == torch.float32):
            revision = "main"
        elif (dtype == torch.float16):
            revision = "float16"
            
        snapshot_download(
            repo_id="google/paligemma-3b-pt-224",
            local_dir=paligemma_path,
            allow_patterns=["*.safetensors", "*.json"],
            revision=revision
        )

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(paligemma_path, "*.safetensors"))
    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if (key.startswith("language_model")):
                    tensors[key] = f.get_tensor(key)

    # Create the model using the configuration
    model = CNNGemmaForConditionalGeneration(config).to(dtype).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)