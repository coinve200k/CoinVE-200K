#!/bin/bash
# Evaluate CoinVE-Bench edited videos with the public Google Gemini API.
#
# Prerequisites:
#   1. export GEMINI_API_KEY=your_key_here
#   2. Place edited videos in GEN_VIDEO_DIR, each named {id}.mp4 where {id} is
#      the integer case id from checklist_json/coinve-bench-361-checklist.json
#      (e.g. 0.mp4, 1.mp4, ..., 360.mp4).
#   3. Activate the inference environment (torch2.8-flash3) so that the
#      google-genai SDK and tqdm are available.
set -euo pipefail

# ─── Auth ────────────────────────────────────────────────────────────
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "[error] GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY=your_key"
    exit 1
fi

# ─── Paths ───────────────────────────────────────────────────────────
DIR=$(cd -- "$(dirname -- "$(realpath "$0")")" &>/dev/null && pwd)
INPUT="${DIR}/checklist_json/coinve-bench-361-checklist.json"
PROMPT_DIR="${DIR}/system_prompts"

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# EDIT THESE TWO VARIABLES FOR YOUR RUN
# GEN_VIDEO_DIR: flat directory containing {id}.mp4 edited videos.
# OUTPUT_NAME:   subdirectory under results/ for this evaluation run.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
GEN_VIDEO_DIR="${DIR}/edited_videos/your_model"
OUTPUT_NAME="your_model"

# ─── Eval options ────────────────────────────────────────────────────
WORKERS=8
MODEL="gemini-3.6-flash"
GEN_VIDEO_EXT=".mp4"
OUT_DIR="${DIR}/results/${OUTPUT_NAME}"

# ─── Run ─────────────────────────────────────────────────────────────
mkdir -p "${OUT_DIR}"
cd "${DIR}"

echo "[eval] input         = ${INPUT}"
echo "[eval] gen_video_dir = ${GEN_VIDEO_DIR}"
echo "[eval] out_dir       = ${OUT_DIR}"
echo "[eval] model         = ${MODEL}"
echo "[eval] workers       = ${WORKERS}"

python eval_coinbench_gemini_public.py \
    --input "${INPUT}" \
    --gen-video-dir "${GEN_VIDEO_DIR}" \
    --gen-video-ext "${GEN_VIDEO_EXT}" \
    --prompt-dir "${PROMPT_DIR}" \
    --out-dir "${OUT_DIR}" \
    --workers "${WORKERS}" \
    --model "${MODEL}" \
    --timestamp \
    2>&1 | tee "${OUT_DIR}/eval.log"
