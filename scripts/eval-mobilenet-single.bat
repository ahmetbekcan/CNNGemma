@echo off
set SCRIPT=src/eval.py
set CONFIG_PATH=configs/mobilenet_single.json
set WEIGHTS_PATH=weights/mobilenet_single
set DATASET_PATH=../dataset/RISCM
set OUTPUT_DIR=evaluate/mobilenet_single
set HUB_ID=ahmetbekcan/CNNGemma-MobileNet-Single-224
set LOG_WANDB=true
set EVAL_PALIGEMMA=false
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
