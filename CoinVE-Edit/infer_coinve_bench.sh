#!/bin/bash
# Multi-GPU residual-attention composite inference with per-instruction MaskHead prediction.
# Saves triptych videos [src | mask overlay | edited] and standalone tgt_videos/<id>.mp4.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_ALGO=TREE,RING
export NCCL_PROTO=SIMPLE,LL,LL128
export NCCL_BUFFSIZE=8388608
export PYTHONWARNINGS="ignore::Warning"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

export SHELL_FILE_REAL_PATH=$(realpath "$0")
export DIR=$(cd -- "$(dirname -- "${SHELL_FILE_REAL_PATH}")" &>/dev/null && pwd)
export CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# EDIT THESE PATHS BEFORE RUNNING.
#
# EVAL_PATH        : evaluation metadata file (.json or .jsonl).
#                    For CoinVE-Bench, point this at the checklist JSON:
#                    CoinVE-Bench/checklist_json/coinve-bench-361-checklist.json
#                    Only `src_video` and `instruction` fields are required.
#                    Set --data_root below to the CoinVE-Bench root so that
#                    relative paths like `src_videos/xxx.mp4` resolve correctly.
# DATA_ROOT        : root directory for resolving relative video paths in the
#                    metadata (e.g. the CoinVE-Bench dataset directory).
# COMPOSITE_CKPT   : path to the CoinVE-Edit .safetensors checkpoint.
# LOCAL_MODEL_PATH : root directory containing `Wan-AI/Wan2.1-T2V-14B/`
#                    (download from https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)
# MLLM_MODEL       : path to the Qwen3-VL-8B-Instruct checkpoint directory
#                    (download from https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
# OUTPUT_BASE_DIR  : root directory for this run's outputs (videos/, config/, log.txt).
#                    Defaults to ${DIR}/ckpt/<EXP_NAME>/<timestamp>.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
EVAL_PATH="/path/to/CoinVE-Bench/checklist_json/coinve-bench-361-checklist.json"
DATA_ROOT="/path/to/CoinVE-Bench"
COMPOSITE_CKPT="/path/to/FireCRT/CoinVE-Edit/coinve_edit_composite_vllm256_dit128.safetensors"
LOCAL_MODEL_PATH="/path/to/base_models"
MLLM_MODEL="/path/to/Qwen3-VL-8B-Instruct"

EXP_NAME="coinve_bench_infer"
OUTPUT_BASE_DIR="${DIR}/outputs
/${EXP_NAME}/${CURRENT_TIME}"

NUM_GPUS=4
MAIN_PROCESS_IP=127.0.0.1
NUM_MACHINES=1
NUM_PROCESSES=${NUM_GPUS}

if [[ ! -f "${COMPOSITE_CKPT}" ]]; then
  echo "Missing composite checkpoint: ${COMPOSITE_CKPT}"
  exit 1
fi
if [[ ! -f "${EVAL_PATH}" ]]; then
  echo "Missing eval metadata: ${EVAL_PATH}"
  exit 1
fi

mkdir -p "${OUTPUT_BASE_DIR}/config"
sed -e "s/\${MAIN_PROCESS_IP}/${MAIN_PROCESS_IP}/g" \
    -e "s/\${NUM_MACHINES}/${NUM_MACHINES}/g" \
    -e "s/\${NUM_PROCESSES}/${NUM_PROCESSES}/g" \
    "${DIR}/config/accelerate_ddp_TEMPLATE.yaml" > "${OUTPUT_BASE_DIR}/config/accelerate_ddp_runtime.yaml"

cp "${SHELL_FILE_REAL_PATH}" "${OUTPUT_BASE_DIR}/"

cat <<EOF
===============================================================
OUTPUT          : ${OUTPUT_BASE_DIR}
EVAL_PATH       : ${EVAL_PATH}
COMPOSITE_CKPT  : ${COMPOSITE_CKPT}
LOCAL_MODEL_PATH: ${LOCAL_MODEL_PATH}
MLLM_MODEL      : ${MLLM_MODEL}
CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}
NUM_PROCESSES   : ${NUM_PROCESSES}
===============================================================
EOF

cd "${DIR}"
accelerate launch --config_file "${OUTPUT_BASE_DIR}/config/accelerate_ddp_runtime.yaml" \
  infer_coinve_bench.py \
  --eval_json_path "${EVAL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_BASE_DIR}/videos" \
  --composite_checkpoint "${COMPOSITE_CKPT}" \
  --local_model_path "${LOCAL_MODEL_PATH}" \
  --mllm_model "${MLLM_MODEL}" \
  --eval_max_pixels 921600 \
  --eval_max_frame 49 \
  --seed 0 \
  --num_inference_steps 50 \
  --skip_existing \
  2>&1 | tee "${OUTPUT_BASE_DIR}/log.txt"

echo "Done. Outputs at ${OUTPUT_BASE_DIR}/videos"
echo "Standalone edited videos for evaluation: ${OUTPUT_BASE_DIR}/videos/tgt_videos/"
