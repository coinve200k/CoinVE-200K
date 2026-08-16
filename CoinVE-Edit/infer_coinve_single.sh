#!/bin/bash
# Single-video CoinVE composite editing on GPU 0.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

DIR=$(cd -- "$(dirname -- "$(realpath "$0")")" &>/dev/null && pwd)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# BASE MODEL PATHS — please specify the local paths to the base models.
# LOCAL_MODEL_PATH : root directory containing `Wan-AI/Wan2.1-T2V-14B/`
#                    (downloaded from https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)
# MLLM_MODEL       : path to the Qwen3-VL-8B-Instruct checkpoint directory
#                    (downloaded from https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
LOCAL_MODEL_PATH="/path/to/base_models"
MLLM_MODEL="/path/to/Qwen3-VL-8B-Instruct"

SRC_VIDEO="${DIR}/data/src_video.mp4"
PROMPTS=(
  "Restyle the man centered in the frame into a cel-shaded anime style."
  "Add a potted green plant near the vertical black cable or rod structure visible on the left side background."
  "Remove the hanging pendant light fixture visible on the right side background."
  "Replace the multi-colored horizontal wooden plank wall behind the man with a solid white brick wall."
)
COMPOSITE_CKPT="/path/to/FireCRT/CoinVE-Edit/coinve_edit_composite_vllm256_dit128.safetensors"
OUTPUT_DIR="${DIR}/outputs/$(date '+%Y.%m.%d-%H.%M.%S')"

mkdir -p "${OUTPUT_DIR}"
cd "${DIR}"
python infer_coinve_single.py \
  --src_video "${SRC_VIDEO}" \
  --prompts "${PROMPTS[@]}" \
  --composite_checkpoint "${COMPOSITE_CKPT}" \
  --local_model_path "${LOCAL_MODEL_PATH}" \
  --mllm_model "${MLLM_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --eval_max_pixels 921600 \
  --eval_max_frame 49 \
  --seed 0 \
  --num_inference_steps 50 \
  --show_progress \
  --save_instruction_masks \
  2>&1 | tee "${OUTPUT_DIR}/log.txt"
