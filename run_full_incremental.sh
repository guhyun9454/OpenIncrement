#!/usr/bin/env bash

# Usage:
#   bash run_full_incremental.sh \
#     --dataset cifar100 \
#     --model resnet18 \
#     --data_folder ../datasets \
#     --base_epochs 600 \
#     --inc_epochs 600 \
#     --batch_size 512 \
#     --learning_rate 0.001 \
#     --temp 0.05 \
#     --alfa 0.2 \
#     --fixed_memory 2000 \
#     --init_classes 10 \
#     --total_classes 100

set -euo pipefail
cd "$(dirname "$0")"

# Defaults
DATASET="mnist"
MODEL="mlp"
DATA_FOLDER="./datasets"
BASE_EPOCHS=200
INC_EPOCHS=200
BATCH_SIZE=256
LR=0.001
TEMP=0.05
ALFA=0.2
FIXED_MEM=500
INIT_CLASSES=8
TOTAL_CLASSES=10
PRINT_FREQ=10
SAVE_FREQ=50

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset) DATASET="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --data_folder) DATA_FOLDER="$2"; shift 2;;
    --base_epochs) BASE_EPOCHS="$2"; shift 2;;
    --inc_epochs) INC_EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --learning_rate) LR="$2"; shift 2;;
    --temp) TEMP="$2"; shift 2;;
    --alfa) ALFA="$2"; shift 2;;
    --fixed_memory) FIXED_MEM="$2"; shift 2;;
    --init_classes) INIT_CLASSES="$2"; shift 2;;
    --total_classes) TOTAL_CLASSES="$2"; shift 2;;
    --print_freq) PRINT_FREQ="$2"; shift 2;;
    --save_freq) SAVE_FREQ="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# 1) Base training (0..INIT_CLASSES-1)
python3 main_supcon.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --data_folder ${DATA_FOLDER} \
  --learning_rate ${LR} \
  --temp ${TEMP} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${BASE_EPOCHS} \
  --num_classes ${INIT_CLASSES} \
  --print_freq ${PRINT_FREQ} \
  --save_freq ${SAVE_FREQ} \
  --alfa ${ALFA} \
  --fixed_memory ${FIXED_MEM}

# 2) Incremental steps (INIT_CLASSES -> TOTAL_CLASSES by step of 10 or the remainder)
current=${INIT_CLASSES}
while [[ ${current} -lt ${TOTAL_CLASSES} ]]; do
  next=$(( current + 10 ))
  if [[ ${next} -gt ${TOTAL_CLASSES} ]]; then
    next=${TOTAL_CLASSES}
  fi
  echo "[Incremental] ${current} -> ${next}"
  python3 main_supcon_incremental.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --data_folder ${DATA_FOLDER} \
    --learning_rate ${LR} \
    --temp ${TEMP} \
    --alfa ${ALFA} \
    --batch_size ${BATCH_SIZE} \
    --fixed_memory ${FIXED_MEM} \
    --num_init_classes ${current} \
    --num_classes ${next} \
    --epochs ${INC_EPOCHS} \
    --img_size $([[ "${DATASET}" == "mnist" ]] && echo 28 || echo 32) \
    --print_freq ${PRINT_FREQ} \
    --save_freq ${SAVE_FREQ}
  current=${next}
  sleep 1
done

echo "All steps done."
