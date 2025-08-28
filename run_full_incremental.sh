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
MODEL="resnet18"
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
LINEAR_EPOCHS=50
LINEAR_LR=0.1

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
    --base_epochs ${BASE_EPOCHS} \
    --num_init_classes ${current} \
    --num_classes ${next} \
    --epochs ${INC_EPOCHS} \
    --img_size $([[ "${DATASET}" == "mnist" ]] && echo 28 || echo 32) \
    --print_freq ${PRINT_FREQ} \
    --save_freq ${SAVE_FREQ}
  current=${next}

  # 3) Linear classifier training for current stage
  python3 main_linear.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --data_folder ${DATA_FOLDER} \
    --epochs ${LINEAR_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LINEAR_LR} \
    --num_classes ${next} \
    --temp ${TEMP} --alfa ${ALFA} --fixed_memory ${FIXED_MEM} \
    --base_epochs ${BASE_EPOCHS} \
    --encoder_epochs ${INC_EPOCHS} \
    --encoder_learning_rate ${LR}

  # 3-1) Classifier evaluation on inlier for current stage
  python3 test_linear.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --data_folder ${DATA_FOLDER} \
    --num_classes ${next}

  # 4) OSR evaluation for current stage
  python3 knn.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --data_folder ${DATA_FOLDER} \
    --num_classes ${next} \
    --epochs ${INC_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --temp ${TEMP} --alfa ${ALFA} --fixed_memory ${FIXED_MEM}
  sleep 1
done

# 3) Linear classifier training (freeze encoder)
python3 main_linear.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --data_folder ${DATA_FOLDER} \
  --epochs ${LINEAR_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LINEAR_LR} \
  --num_classes ${TOTAL_CLASSES} \
  --temp ${TEMP} --alfa ${ALFA} --fixed_memory ${FIXED_MEM} \
  --base_epochs ${BASE_EPOCHS}

# 3-1) Classifier evaluation on inlier
python3 test_linear.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --data_folder ${DATA_FOLDER} \
  --num_classes ${TOTAL_CLASSES}

# 4) OSR evaluation (KNN-style)
python3 knn.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --data_folder ${DATA_FOLDER} \
  --num_classes ${TOTAL_CLASSES} \
  --epochs ${INC_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --temp ${TEMP} --alfa ${ALFA} --fixed_memory ${FIXED_MEM}

echo "All steps done."
