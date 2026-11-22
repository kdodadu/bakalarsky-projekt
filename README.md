# Fine-Tuning klasifikačných modelov na identifikáciu multimodálnych dezinformačných tokov na digitálnych platformách

*Fine-Tuning Classification Models for Identifying Multimodal Disinformation Flows on Digital Platforms* — bachelor's project.

The project evaluates vision-language models on image–caption consistency classification. Given an image and a caption, a model predicts `MATCH` or `MISMATCH`.

The current implementation provides zero-shot baselines for three models on two datasets. No task-specific fine-tuning is applied in the reported experiments. The task concerns image–caption consistency, rather than independent verification of a news claim.

## Models

| Model | Developer | Checkpoint |
|---|---|---|
| LFM3-VL-2B | Liquid AI | [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) |
| Qwen3.5-2B | Qwen / Alibaba | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| SmolVLM2-2.2B-Instruct | Hugging Face | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) |

All models run locally through llama.cpp using Q4_K_M GGUF weights and their corresponding vision components (`mmproj`). Exact weight revisions, hashes and inference settings are recorded in the configuration files.

## Data

### NewsCLIPpings

[NewsCLIPpings](https://github.com/g-luo/news_clippings) contains original and automatically constructed mismatched image–caption pairs based on [VisualNews](https://github.com/FuxiaoLiu/VisualNews-Repository).

This project evaluates the official `merged_balanced` test split:

- **7,264 pairs**
- **3,632 MATCH:** original image–caption pairs
- **3,632 MISMATCH:** constructed mismatched pairs

Captions and images are retrieved separately using the annotation's caption and image identifiers.

### VERITE

[VERITE](https://github.com/stevejpapad/image-text-verification) is used as an external evaluation dataset containing true, miscaptioned and out-of-context image–caption pairs.

The supplied dataset version contains **1,001 records**. This project uses an **adapted binary evaluation**:

| Original category | Records | Binary label |
|---|---:|---|
| true | 338 | MATCH |
| miscaptioned | 338 | MISMATCH |
| out-of-context | 325 | MISMATCH |
| **Total** | **1,001** | **338 MATCH / 663 MISMATCH** |

Original categories are preserved for separate reporting. This mapping differs from the original three-class task and changes the class balance. Results should not be compared directly with three-class evaluation scores.

A constant `MISMATCH` prediction achieves **66.23% accuracy** on this binary mapping. No prompts, thresholds or model parameters are tuned using VERITE labels.

Dataset files are not distributed with this repository. Obtain them from their respective sources and follow the applicable access and usage conditions.

## Evaluation and results

All models receive the same instruction from [`configs/prompt.txt`](configs/prompt.txt), an image and its caption. Each pair is evaluated independently, without conversation history or labelled examples.

The shared pipeline handles data preparation, inference, resumable prediction storage and automatic reporting.

All attempted records count toward evaluation. Request errors (`ERROR`) and unparseable responses (`INVALID`) count as incorrect predictions and false negatives for the true class. Macro-F1 averages the F1 scores of MATCH and MISMATCH. Valid response rate measures the proportion of requests producing a valid class label.

All six runs are complete. Precision, recall and F1 are rounded to four decimal places; support is the number of ground-truth examples in each class.

### NewsCLIPpings — merged_balanced test

| Model | Evaluated | Accuracy | Macro-F1 | Valid responses | ERROR | INVALID |
|---|---:|---:|---:|---:|---:|---:|
| LFM3-VL-2B | 7,264/7,264 | 58.16% | 0.5260 | 99.94% | 4 | 0 |
| Qwen3.5-2B | 7,264/7,264 | **67.99%** | **0.6800** | 99.94% | 4 | 0 |
| SmolVLM2-2.2B-Instruct | 7,264/7,264 | 50.30% | 0.3423 | 99.94% | 4 | 0 |

The four errors concern the same JPEG inputs rejected by the server's image decoder. They remain included in the metrics.

**Class-level metrics**

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| LFM3-VL-2B | MATCH | 0.7619 | 0.2387 | 0.3635 | 3,632 |
| LFM3-VL-2B | MISMATCH | 0.5485 | 0.9246 | 0.6885 | 3,632 |
| Qwen3.5-2B | MATCH | 0.6744 | 0.6974 | 0.6857 | 3,632 |
| Qwen3.5-2B | MISMATCH | 0.6866 | 0.6624 | 0.6743 | 3,632 |
| SmolVLM2-2.2B-Instruct | MATCH | 0.5017 | 0.9975 | 0.6676 | 3,632 |
| SmolVLM2-2.2B-Instruct | MISMATCH | 0.7949 | 0.0085 | 0.0169 | 3,632 |

**Confusion matrices, including failures**

Rows represent ground-truth classes; prediction columns contain counts.

| Model | True class | Predicted MATCH | Predicted MISMATCH | INVALID | ERROR |
|---|---|---:|---:|---:|---:|
| LFM3-VL-2B | MATCH | 867 | 2,764 | 0 | 1 |
| LFM3-VL-2B | MISMATCH | 271 | 3,358 | 0 | 3 |
| Qwen3.5-2B | MATCH | 2,533 | 1,098 | 0 | 1 |
| Qwen3.5-2B | MISMATCH | 1,223 | 2,406 | 0 | 3 |
| SmolVLM2-2.2B-Instruct | MATCH | 3,623 | 8 | 0 | 1 |
| SmolVLM2-2.2B-Instruct | MISMATCH | 3,598 | 31 | 0 | 3 |

Qwen achieves the highest accuracy and macro-F1. LFM favours MISMATCH, while SmolVLM2 predicts MATCH for almost all successfully processed inputs.

Full reports:
[LFM](reports/newsclippings/lfm3_vl_2b/metrics.json) ·
[Qwen](reports/newsclippings/qwen35_2b/metrics.json) ·
[SmolVLM2](reports/newsclippings/smolvlm2_2_2b/metrics.json) ·
[Comparison](reports/newsclippings/model_comparison.csv)

### VERITE — adapted binary evaluation

| Model | Evaluated | Accuracy | Macro-F1 | Valid responses | ERROR | INVALID |
|---|---:|---:|---:|---:|---:|---:|
| LFM3-VL-2B | 1,001/1,001 | **64.34%** | 0.5672 | 100% | 0 | 0 |
| Qwen3.5-2B | 1,001/1,001 | 61.44% | **0.6104** | 100% | 0 | 0 |
| SmolVLM2-2.2B-Instruct | 1,001/1,001 | 34.97% | 0.2725 | 100% | 0 | 0 |

All models score below the **66.23% accuracy** achieved by always predicting MISMATCH. Accuracy should therefore be interpreted alongside macro-F1 and class-level results.

**Class-level metrics**

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| LFM3-VL-2B | MATCH | 0.4609 | 0.3314 | 0.3855 | 338 |
| LFM3-VL-2B | MISMATCH | 0.7018 | 0.8024 | 0.7488 | 663 |
| Qwen3.5-2B | MATCH | 0.4573 | 0.7604 | 0.5711 | 338 |
| Qwen3.5-2B | MISMATCH | 0.8155 | 0.5400 | 0.6497 | 663 |
| SmolVLM2-2.2B-Instruct | MATCH | 0.3418 | 1.0000 | 0.5094 | 338 |
| SmolVLM2-2.2B-Instruct | MISMATCH | 1.0000 | 0.0181 | 0.0356 | 663 |

**Confusion matrices, including failures**

Rows represent ground-truth classes; prediction columns contain counts.

| Model | True class | Predicted MATCH | Predicted MISMATCH | INVALID | ERROR |
|---|---|---:|---:|---:|---:|
| LFM3-VL-2B | MATCH | 112 | 226 | 0 | 0 |
| LFM3-VL-2B | MISMATCH | 131 | 532 | 0 | 0 |
| Qwen3.5-2B | MATCH | 257 | 81 | 0 | 0 |
| Qwen3.5-2B | MISMATCH | 305 | 358 | 0 | 0 |
| SmolVLM2-2.2B-Instruct | MATCH | 338 | 0 | 0 | 0 |
| SmolVLM2-2.2B-Instruct | MISMATCH | 651 | 12 | 0 | 0 |

**Results by original category**

Each cell shows correct binary predictions / category size and the corresponding binary accuracy. These are not three-class predictions.

| Model | true | miscaptioned | out-of-context |
|---|---:|---:|---:|
| LFM3-VL-2B | 112/338 (33.14%) | 249/338 (73.67%) | 283/325 (87.08%) |
| Qwen3.5-2B | 257/338 (76.04%) | 169/338 (50.00%) | 189/325 (58.15%) |
| SmolVLM2-2.2B-Instruct | 338/338 (100%) | 7/338 (2.07%) | 5/325 (1.54%) |

LFM detects more mismatched pairs but rejects many true pairs. Qwen achieves the highest macro-F1. SmolVLM2 predicts MATCH for 989 of 1,001 inputs; its perfect MISMATCH precision is based on only 12 predictions, with a recall of 1.81%.

Full reports:
[LFM](reports/verite/lfm3_vl_2b/metrics.json) ·
[Qwen](reports/verite/qwen35_2b/metrics.json) ·
[SmolVLM2](reports/verite/smolvlm2_2_2b/metrics.json) ·
[Comparison](reports/verite/model_comparison.csv)

## Repository structure

```text
configs/
  prompt.txt
  newsclippings/           model configurations
  verite/                  model configurations
scripts/
  baseline.py              shared command-line entry point
src/
  multimodal_baseline/     data loading, inference, evaluation and storage
tests/                     automated tests
reports/
  newsclippings/           aggregate metrics and model comparison
  verite/                  aggregate metrics and model comparison

data/                      local datasets; not versioned
results/                   local predictions and run metadata; not versioned
logs/                      local server logs; not versioned
```

## Requirements and usage

Python 3.13+ and llama.cpp are required. Exact model files and server settings are specified in `configs/<dataset>/<model_id>.json`.

The baseline uses only the Python standard library. Dependencies for the separate training environment are listed in `requirements.txt` and can be installed with:

```powershell
pip install -r requirements.txt
```

After obtaining the dataset and model files, prepare the data and start the corresponding server using the configuration's `launch_command`.

```powershell
# Prepare the dataset
.\.venv\Scripts\python.exe scripts/baseline.py --dataset verite --model qwen35_2b prepare

# Run the evaluation after starting the server
.\.venv\Scripts\python.exe scripts/baseline.py --dataset verite --model qwen35_2b run

# Inspect progress
.\.venv\Scripts\python.exe scripts/baseline.py --dataset verite --model qwen35_2b status

# Resume an interrupted run
.\.venv\Scripts\python.exe scripts/baseline.py --dataset verite --model qwen35_2b run --resume
```

Supported datasets: `newsclippings`, `verite`.

Supported models: `lfm3_vl_2b`, `qwen35_2b`, `smolvlm2_2_2b`.

Reports are updated automatically. To rebuild them from saved predictions:

```powershell
.\.venv\Scripts\python.exe scripts/baseline.py --dataset verite compare
```