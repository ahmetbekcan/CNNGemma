@echo off
set SCRIPT=src\train.py
set CONFIG_PATH=configs\mobilenet_single.json
set WEIGHTS_PATH=weights\pretrained\paligemma
set DATASET_PATH=..\dataset\RISCM
set OUTPUT_DIR=weights\mobilenet_single
set HUB_ID=ahmetbekcan/CNNGemma-MobileNet-Single-224

python %SCRIPT% ^
  --config %CONFIG_PATH% ^
  --weights %WEIGHTS_PATH% ^
  --dataset %DATASET_PATH% ^
  --output_dir %OUTPUT_DIR% ^
  --hub_id %HUB_ID%
