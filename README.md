# Fine-Tuning klasifikačných modelov na identifikáciu multimodálnych dezinformačných tokov na digitálnych platformách

*Fine-Tuning Classification Models to Identify Multimodal Disinformation Flows on Digital Platforms* — bachelor's project at the Technical University of Košice.

This repository contains the baseline evaluation before fine-tuning. Given a news photograph and a caption, the model predicts whether they form the original pair (`MATCH`) or were taken from different news items (`MISMATCH`).

## Model and data

The baseline runs [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) locally with `llama.cpp`, using [Q4_K_M quantized weights](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF) and a fixed [classification prompt](configs/prompt.txt).

Evaluation uses the official `merged_balanced` test split of [NewsCLIPpings](https://github.com/g-luo/news_clippings). Its annotations specify which [VisualNews](https://github.com/FuxiaoLiu/VisualNews-Repository) photograph and caption to pair. Model weights and datasets are downloaded separately.

## Baseline results

| Test pairs | Accuracy | Macro-F1 | Invalid responses | Request errors |
|---:|---:|---:|---:|---:|
| 7,264 | 58.18% | 0.5258 | 0 | 0 |

The model often rejected original image–caption pairs. The [full metric summary](reports/baseline_metrics.json) includes the confusion matrix. These results measure how well the model detects a swapped photograph, not whether the underlying news story is true.

## Repository structure

| Path | Purpose |
|---|---|
| `configs/` | Evaluation settings and classification prompt |
| `scripts/baseline.py` | Command-line entry point for the baseline |
| `src/newsclip_pilot/` | Data preparation, model requests, scoring and run storage |
| `tests/test_baseline.py` | Tests for the evaluation pipeline |
| `reports/baseline_metrics.json` | Results from the completed test run |

## Requirements

Python and `llama.cpp`. The baseline was run on Windows 11 with an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```