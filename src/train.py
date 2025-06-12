import torch
import PIL
import wandb
from cnn_gemma import CNNGemmaConfig
from processing_cnngemma import CNNGemmaProcessor
from utils import load_pretrained_model
import json
from pathlib import Path
from datasets import load_dataset
from transformers import Trainer
from transformers import TrainingArguments
import argparse
import random
from transformers import set_seed
import random
import os

set_seed(2299436)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Relative or absolute path to the model config JSON file.")
parser.add_argument("--weights", type=str, default="weights/pretrained/paligemma", help="Relative or absolute path to the pretrained PaliGemma weights.")
parser.add_argument("--dataset", type=str, default="../dataset/RISCM", help="""Relative or absolute path to the dataset folder.
                    Images should be placed under folder \"resized\" and csv files should be in given folder""")
parser.add_argument("--output_dir", type=str, default="weights/output/finetune_test", help="Relative or absolute path to save the finetuned model.")
parser.add_argument("--hub_id", type=str, default="", help="Hugging Face repository id")
args_cli = parser.parse_args()

def resolve_path(user_path: str) -> Path:
    p = Path(user_path)
    return p if p.is_absolute() else (Path(__file__).parent.resolve() / p)

absolute_config_path = resolve_path(args_cli.config)
absolute_weight_path = resolve_path(args_cli.weights)
absolute_dataset_path = resolve_path(args_cli.dataset)

if args_cli.output_dir == "auto":
    config_stem = absolute_config_path.stem
    absolute_weight_output_dir = Path("weights/output") / config_stem
else:
    absolute_weight_output_dir = resolve_path(args_cli.output_dir)

if not absolute_config_path.is_file():
    raise FileNotFoundError(f"Config file not found: {absolute_config_path}")

if not absolute_dataset_path.exists():
    raise FileNotFoundError(f"Dataset path not found: {absolute_dataset_path}")

with open(absolute_config_path, "r") as f:
    model_config_file = json.load(f)

device = "cuda"
dtype = torch.bfloat16
config = CNNGemmaConfig(**model_config_file)
print("Loading pretrained model...")
model, tokenizer = load_pretrained_model(paligemma_path=absolute_weight_path, device=device, config=config, dtype=dtype)

processor = CNNGemmaProcessor(tokenizer=tokenizer,num_image_tokens=model.config.vision_config.num_image_tokens, image_encoder_type=model.config.vision_config.architecture)

def collate_fn(examples):
      texts = ["caption en" for example in examples]
      labels = [random.choice([ex[f"caption_{i}"] for i in range(1, 6)]) for ex in examples]
      images = [PIL.Image.open(absolute_dataset_path / "resized" / example["image"]).convert("RGB") for example in examples]
      inputs = processor(text=texts, images=images, labels=labels, padding="longest")
      inputs = {k: v.to(device) for k, v in inputs.items()}
      return inputs

for param in model.language_model.parameters():
    param.requires_grad = False

print("Loading dataset...")
captions_csv_path = str(absolute_dataset_path / "captions.csv")
dataset = load_dataset("csv", data_files=captions_csv_path, split="train")

train_dataset = dataset.filter(lambda x: x["split"] == "train")
val_dataset = dataset.filter(lambda x: x["split"] == "val")

total_train_samples = len(train_dataset)
batch_size = 1
gradient_accumulation_steps = 4
steps_per_epoch = total_train_samples // (batch_size * gradient_accumulation_steps)
save_steps = steps_per_epoch // 1

wandb.login()

os.environ["WANDB_PROJECT"] = "Final-Project"

push_to_hub = True
if (args_cli.hub_id == ""):
    push_to_hub = False

args = TrainingArguments(
    output_dir=absolute_weight_output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_ratio=0.05,
    learning_rate=1e-4,
    weight_decay=1e-4,
    adam_beta2=0.999,
    lr_scheduler_type="cosine",
    logging_steps=10,
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=3,
    save_safetensors=True,
    bf16=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    push_to_hub=push_to_hub,
    hub_model_id=args_cli.hub_id,
    hub_strategy="end",
    report_to="wandb",
    run_name=f"finetune-{absolute_config_path.stem}"
)

trainer = Trainer(
      model=model,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      data_collator=collate_fn,
      args=args
)
trainer.train()

if (push_to_hub):
    trainer.push_to_hub()
