from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
from mobile_gemma import MobileGemmaForConditionalGeneration, MobileGemmaConfig
from huggingface_hub import snapshot_download
import torch

def download_model(model_path: str):

    snapshot_download(
        repo_id="ahmetbekcan/mobilegemma-3b-pt-224",
        local_dir=model_path,
        ignore_patterns="*.gitattributes"
    )

    return

def load_hf_model(model_path: str, device: str) -> Tuple[MobileGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", padding_side="right")
    assert tokenizer.padding_side == "right"

    os.makedirs(model_path, exist_ok=True)

    #download safetensors and config.json if not downloaded
    if not any(
        f.endswith('.safetensors') for f in os.listdir(model_path)
        if os.path.isfile(os.path.join(model_path, f))
    ):
        download_model(model_path)

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
        config = MobileGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = MobileGemmaForConditionalGeneration(config).to(torch.bfloat16).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)