@echo off
set SCRIPT=src/eval.py
set CONFIG_PATH=paligemma
set WEIGHTS_PATH=weights/pretrained/paligemma
set DATASET_PATH=../dataset/RISCM
set HUB_ID=google/paligemma-3b-mix-224
set OUTPUT_DIR=evaluate/paligemma
set LOG_WANDB=true
set EVAL_PALIGEMMA=true
set EVAL_PRETRAINED=false

python %SCRIPT% ^
  --config %CONFIG_PATH% ^
  --weight_dir %WEIGHTS_PATH% ^
  --hub_id %HUB_ID% ^
  --log_to_wandb %LOG_WANDB% ^
  --dataset %DATASET_PATH% ^
  --output_dir %OUTPUT_DIR% ^
  --eval_paligemma %EVAL_PALIGEMMA% ^
  --eval_pretrained %EVAL_PRETRAINED%
