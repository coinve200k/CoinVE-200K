# CoinVE-Bench: A Benchmark for Compositional Instruction-Guided Video Editing

<div align="center">

  <a href="https://huggingface.co/datasets/FireCRT/CoinVE-Bench"><img src="https://img.shields.io/static/v1?label=CoinVE-Bench&message=HuggingFace&color=yellow"></a> &ensp;

</div>

---

## ✨ Highlights

- **Compositional Editing Benchmark**: CoinVE-Bench is a benchmark dedicated to compositional (multi-instruction) video editing evaluation, featuring 361 test cases with 2–5 instructions per sample.
- **Region-Aware Evaluation**: Per-instruction masks enable fine-grained checks of whether each edit is confined to the correct region.
- **Multi-Dimensional Metrics**: Comprehensive evaluation across editing accuracy, physical naturalness, and semantic preservation.
- **VLLM-Based Scoring**: Leveraging vision-language models (e.g., Gemini) for human-aligned assessment.


## 🌍 Introduction

Existing video editing benchmarks (e.g., ReCo-Bench, OpenVE-Bench) primarily evaluate **single-instruction editing**, where each test sample contains only one editing operation. However, real-world video editing scenarios often require **compositional instructions** — applying multiple edits (e.g., replacing the subject, adding an object, and changing the background) to the same video simultaneously.

**CoinVE-Bench** is a dedicated benchmark for evaluating compositional instruction-guided video editing. It contains diverse multi-instruction test cases with per-instruction region masks, enabling fine-grained evaluation of whether each instruction is correctly executed in its target region.

**Key features:**
- **Compositional Test Cases**: Each sample contains 2–5 editing instructions covering different operations and regions.
- **Region-Aware Evaluation**: Per-instruction masks enable region-specific fidelity checks.
- **Multi-Dimensional Metrics**: Evaluation across editing accuracy, physical naturalness, and semantic preservation.
- **VLLM-Based Scoring**: Leveraging vision-language models (e.g., Gemini) for comprehensive and human-aligned assessment.


## 📊 Benchmark Statistics

| Metric | Value |
|--------|-------|
| Total test samples | 361 |
| Instructions per sample | 2–5 |
| Checklist questions | 4,131 |
| Evaluation dimensions | 3 (Editing Accuracy / Physical Naturalness / Semantic Preservation) |

### Scoring Dimensions & Metrics

All metrics are scored by **Gemini 3.6 Flash** (default; configurable via `--model`) using a checklist-based VLLM protocol.

| Dimension | Metric (Abbr.) | Range | # Questions | Description |
|-----------|-----------------|-------|-------------|-------------|
| **Editing Accuracy** (per instruction) | Semantic Accuracy (SA) | [0, 100] | 1,081 | Correct execution of the intended edit semantics. |
| | Scope Accuracy (SPA) | [0, 100] | 379 | Correct edit localization without leakage or interference. |
| | Editing Persistence (EP) | [0, 100] | 923 | Temporal consistency of the edit throughout the video. |
| **Physical Naturalness** | Appearance Naturalness (AN) | [0, 100] | 356 | Natural blending of lighting, shadows, textures, and style. |
| | Scale Consistency (SC) | [0, 100] | 521 | Plausibility of the edited object's scale and perspective. |
| | Motion Naturalness (MN) | [0, 100] | 447 | Plausibility of motion and physical interactions. |
| **Semantic Preservation** | Content Preservation (CP) | [0, 100] | 424 | Preservation of non-edited regions, objects, and structures. |

**Scoring formulas:**
- ACC-type metrics (SA / SPA / EP / AN / SC / MN): `score = num_correct / num_questions × 100`
- Score-type metric (CP): `score = mean(score) × 10` (each question is scored 0–10, then scaled to [0, 100])
- Dimension score = arithmetic mean of its metric `score_100` values

### Question Types

| Type | # Questions | Videos Uploaded | Answer Format |
|------|-------------|-----------------|---------------|
| Single True/False (Single-TF) | 2,524 | Edited Video only | Yes / No |
| A/B Multiple Choice Question (AB-MCQ) | 609 | Edited Video only | A / B / A and B |
| Dual True/False (Dual-TF) | 574 | Source Video + Edited Video | Yes / No |
| Score Multiple Choice Question (Score-MCQ) | 424 | Source Video + Edited Video | 0–10 score |

## 📥 Download

```bash
# Download CoinVE-Bench from HuggingFace
hf download FireCRT/CoinVE-Bench --repo-type dataset --local-dir ./CoinVE-Bench
```

### Expected Directory Structure

```text
CoinVE-Bench/
├── src_videos/
│   ├── 9CicJDFN1TA_9_0to183.mp4
│   ├── gWG0LiuZnFQ_8_0to137.mp4
│   ├── JwVDrRmyoXc_51_0to172.mp4
│   └── ...
└── checklist_json/
    └── coinve-bench-361-checklist.json
```

Source videos are named `<youtube_id>_<clip_index>_<start>to<end>.mp4` and are referenced by the `src_video` field of each case in the checklist (e.g. `src_videos/9CicJDFN1TA_9_0to183.mp4`).


## 🔧 Evaluation

Evaluation is a two-stage pipeline: (1) run your editing model on the 361 source videos to produce edited videos, (2) score them with the Gemini-based evaluator.

### 1. Run Inference on CoinVE-Bench

Generate one edited video per test case using your model. For CoinVE-Edit, use `infer_coinve_bench.py` in the [`CoinVE-Edit/`](../CoinVE-Edit) directory (see its [README](../CoinVE-Edit/README.md) for arguments).

### 2. Prepare Edited Videos

The evaluator expects **one flat directory** of edited MP4 files with a strict naming convention:

```text
edited_videos/your_model/
├── 0.mp4
├── 1.mp4
├── 2.mp4
└── ...
└── 360.mp4
```

- **Format**: MP4 (`.mp4`). Use `--gen-video-ext` to override if your model outputs a different container.
- **Naming**: `<case_id>.mp4` where `<case_id>` is the integer `id` field of each case in `checklist_json/coinve-bench-361-checklist.json` (range 0–360, **no zero-padding**).
- **Source video pairing**: Video A (original) is read from `case["src_video"]` inside the checklist; you do **not** need to place source videos in this directory.
- **Missing videos**: if a case's edited video is absent, all its questions are recorded with `model_answer = null` and counted as failures in the final score.

> **Note for CoinVE-Edit users**: `infer_coinve_bench.py` saves two artifacts per case — a side-by-side comparison video (`<idx:04d>_K<n>_<prompt>_<src_stem>.mp4`) and a standalone edited video at `tgt_videos/<case_id>.mp4`. The latter is already in the `{id}.mp4` naming convention expected by the evaluator, so you can pass the `tgt_videos/` directory directly as `--gen-video-dir`.

### 3. Run Gemini Evaluation
Set your Gemini API key, edit `eval_coinbench_gemini_public.sh` to point `GEN_VIDEO_DIR` at your edited-video directory, then run:

```bash
export GEMINI_API_KEY=your_key_here
bash eval_coinbench_gemini_public.sh
```

Equivalent direct invocation:

```bash
python eval_coinbench_gemini_public.py \
    --input ./checklist_json/coinve-bench-361-checklist.json \
    --gen-video-dir ./edited_videos/your_model \
    --prompt-dir ./system_prompts \
    --out-dir ./results/your_model \
    --workers 8 \
    --model gemini-3.6-flash \
    --timestamp
```

The script performs everything end-to-end — Gemini calls, per-question scoring, dimension/metric aggregation, and report writing. **No separate aggregation step is needed.**

#### Outputs

Written to `--out-dir` (default `results/<output-name>/`):

| File | Description |
|------|-------------|
| `*_evaluated.jsonl` | Checklist with `model_answer` / `model_reasoning` filled in (resumable) |
| `*_stats.json` | Final scores: 3 dimensions, 7 metrics, `case_all_correct` |
| `*_per_case.json` | Per-case breakdown for fine-grained analysis |
| `*_eval.errors.json` | Failed/retried questions (if any) |

#### Useful Options

| Flag | Purpose |
|------|---------|
| `--only-aggregate` | Skip Gemini calls; recompute scores from an existing `*_evaluated.jsonl` |
| `--overwrite` | Clear previous `model_answer` and rerun every question |
| `--limit N` | Evaluate only the first N cases (quick smoke test) |
| `--max-rounds R` | Max retry rounds for failed/throttled questions (default 10) |
| `--workers W` | Parallel Gemini requests (default 8) |


## 📈 Performance Comparisons

| Model | Overall | Instruction | Temporal | Regional | Visual |
|-------|---------|-------------|----------|----------|--------|
| TBD | TBD | TBD | TBD | TBD | TBD |




## 📜 Citation

If you find CoinVE-Bench useful for your research, please cite our work:

```bibtex
@article{coinve200k,
  title={CoinVE-200K: A Large-Scale High-Quality Dataset for Compositional Instruction-Guided Video Editing},
  author={Long, Fuchen and Wang, Cong and Gao, Zitao and Zhong, Wenhao and Cheng, Yu and Hou, Xiaolu and Li, Yan and Cao, Xiao and Sun, Xinlong and Chen, Xi and Liu, Yu},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```


## ✉️ Contact

For any questions, issues, or collaborations, please feel free to contact longfc.ustc@gmail.com.


## 💖 Acknowledgement

Our benchmark construction and evaluation protocol are inspired by [ReCo-Bench](https://huggingface.co/datasets/HiDream-ai/ReCo-Bench) and [OpenVE-Bench](https://huggingface.co/datasets/Lewandofski/OpenVE-Bench). Thanks to the contributors of all these remarkable projects!
