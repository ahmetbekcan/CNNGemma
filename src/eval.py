import torch
from evaluate import load
import wandb
from cnn_gemma import CNNGemmaConfig
from processing_cnngemma import CNNGemmaProcessor
from utils import load_pretrained_model, load_finetuned_model
import json
from pathlib import Path
from datasets import load_dataset
import argparse
from inference import inference
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

#--------------------------------PARSING ARGUMENTS----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Relative or absolute path to the model config JSON file.")
parser.add_argument("--weight_dir", required=True, type=str, help="Relative or absolute path to the model weights.")
parser.add_argument("--hub_id", required=True, type=str, help="Hugging Face model id of the finetuned model.")
parser.add_argument("--log_to_wandb", type=lambda v: v.lower() == "true", choices=[True, False], default="true", help="Log results to Weights & Biases.")
parser.add_argument("--dataset", type=str, default="../dataset/RISCM", help="Relative or absolute path to the dataset folder. Images should be placed under folder 'resized' and csv files should be in given folder.")
parser.add_argument("--output_dir", type=str, default="evaluate/test", help="Relative or absolute path to save the evaluation results.")
parser.add_argument("--eval_paligemma", type=lambda v: v.lower() == "true", choices=[True, False], default="false", help="Evaluate PaliGemma.")
parser.add_argument("--eval_pretrained", type=lambda v: v.lower() == "true", choices=[True, False], default="false", help="Evaluate pretrained CNNGemma model.")

args_cli = parser.parse_args()

def resolve_path(user_path: str) -> Path:
    p = Path(user_path)
    return p if p.is_absolute() else (Path(__file__).parent.resolve() / p)

absolute_config_path = resolve_path(args_cli.config)
absolute_weight_path = resolve_path(args_cli.weight_dir)
absolute_dataset_path = resolve_path(args_cli.dataset)

if args_cli.output_dir == "auto":
    config_stem = absolute_config_path.stem
    absolute_eval_output_dir = Path("evaluation/output") / config_stem
else:
    absolute_eval_output_dir = resolve_path(args_cli.output_dir)

if not absolute_config_path.is_file():
    raise FileNotFoundError(f"Config file not found: {absolute_config_path}")

if not absolute_dataset_path.exists():
    raise FileNotFoundError(f"Dataset path not found: {absolute_dataset_path}")

with open(absolute_config_path, "r") as f:
    model_config_file = json.load(f)
#--------------------------------PARSING ARGUMENTS----------------------------------

absolute_eval_output_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
config = CNNGemmaConfig(**model_config_file)
print("Loading model...")

if (args_cli.eval_paligemma):
    model_id = "google/paligemma-3b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
    processor = AutoProcessor.from_pretrained(model_id)
elif (args_cli.eval_pretrained):
    model, tokenizer = load_pretrained_model(model_path=absolute_weight_path, device=device, config=config, dtype=dtype)
    processor = CNNGemmaProcessor(tokenizer=tokenizer,num_image_tokens=model.config.vision_config.num_image_tokens, image_encoder_type=model.config.vision_config.architecture)
else:
    model, tokenizer = load_finetuned_model(model_path=absolute_weight_path,hub_id=args_cli.hub_id ,device=device, config=config, dtype=dtype)
    processor = CNNGemmaProcessor(tokenizer=tokenizer,num_image_tokens=model.config.vision_config.num_image_tokens, image_encoder_type=model.config.vision_config.architecture)

parent_path = Path(__file__).parent.resolve()

print("Loading metric functions...")

bleu = load("bleu")
meteor = load("meteor")
rouge = load("rouge")

run_name = absolute_config_path.stem
if (args_cli.eval_paligemma):
    run_name = "paligemma"
elif (args_cli.eval_pretrained):
    run_name += "-pretrained"
else:
    run_name += "-finetuned"

if (args_cli.log_to_wandb):
    print("Initializing w&b")
    wandb.init(project="Final-Project", name=f"eval-{run_name}")
    columns = [
        "image",
        "prediction",
        "caption_1",
        "caption_2",
        "caption_3",
        "caption_4",
        "caption_5",
        "bleu_avg", "bleu_max", "bleu_min",
        "meteor_avg", "meteor_max", "meteor_min",
        "rouge_avg", "rouge_max", "rouge_min",
    ]
    table = wandb.Table(columns=columns)

def evaluate(example):
    image_pth = absolute_dataset_path / "resized" / example["image"]
    prediction = inference(model, processor, prompt="caption en", image_file_path=image_pth, device=device, do_sample=False)

    bleu_scores = []
    meteor_scores = []
    rouge_scores = []

    for key in example:
        if key.startswith("caption"):
            label = example[key]
            bleu_result = bleu.compute(predictions=[prediction], references=[[label]])
            meteor_result = meteor.compute(predictions=[prediction], references=[[label]])
            rouge_result = rouge.compute(predictions=[prediction], references=[[label]])

            bleu_scores.append(bleu_result["bleu"])
            meteor_scores.append(meteor_result["meteor"])
            rouge_scores.append(rouge_result["rougeL"])
    
    bleu_avg = sum(bleu_scores) / len(bleu_scores)
    meteor_avg = sum(meteor_scores) / len(meteor_scores)
    rouge_avg = sum(rouge_scores) / len(rouge_scores)

    if (args_cli.log_to_wandb):
        table.add_data(
            example["image"],
            prediction,
            example["caption_1"],
            example["caption_2"],
            example["caption_3"],
            example["caption_4"],
            example["caption_5"],
            bleu_avg, max(bleu_scores), min(bleu_scores),
            meteor_avg, max(meteor_scores), min(meteor_scores),
            rouge_avg, max(rouge_scores), min(rouge_scores),
        )

    example["prediction"] = prediction
    example["bleu_avg"] = bleu_avg
    example["bleu_max"] = max(bleu_scores)
    example["bleu_min"] = min(bleu_scores)
    example["meteor_avg"] = meteor_avg
    example["meteor_max"] = max(meteor_scores)
    example["meteor_min"] = min(meteor_scores)
    example["rouge_avg"] = rouge_avg
    example["rouge_max"] = max(rouge_scores)
    example["rouge_min"] = min(rouge_scores)

    return example

print("Loading dataset...")
captions_csv_path = str(absolute_dataset_path / "captions.csv")
dataset = load_dataset("csv", data_files=captions_csv_path, split="train")
test_dataset = dataset.filter(lambda x: x["split"] == "test")
keep_columns = [col for col in test_dataset.column_names if col.startswith("image") or col.startswith("caption")]
test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in keep_columns])
test_dataset = test_dataset.map(evaluate)

output_path = absolute_eval_output_dir / "evaluation_results.csv"
test_dataset.to_csv(str(output_path))

if (args_cli.log_to_wandb):
    wandb.log({"evaluation_table": table})

