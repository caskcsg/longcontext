## LongBench-Pro: A More Realistic and Comprehensive Bilingual Long-Context Evaluation Benchmark

This repository contains the evaluation code for **LongBench-Pro**.

<div align="center">

[![HF Dataset](https://img.shields.io/badge/HF-Dataset-yellow?logo=huggingface&logoColor=white)]() &nbsp;&nbsp;
[![Github Code](https://img.shields.io/badge/Github-Code-blue?logo=github&logoColor=white)]() &nbsp;&nbsp;
[![Leaderboard](https://img.shields.io/badge/🏆-Leaderboard-red)]() &nbsp;&nbsp;
[![Paper](https://img.shields.io/badge/📄-Arxiv_Paper-green)]()

</div>

<div align="center">
  <img src="images/performance.png" width="100%"/>
</div>


## Quick Links

  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Inference-Settings](#inference-settings)
  - [Main-Results](#main-results)
  - [Evaluation](#evaluation)


## Overview

Understanding and reasoning long contexts constitute a core capability of modern large language models. However, as model context lengths and long-context reasoning abilities improve rapidly, the development of long-context evaluation benchmarks increasingly lags behind. This lag arises from the trade-off between quality and cost. To address this challenge, we design a novel Human–Model Collaborative Sample Construction strategy that efficiently builds **LongBench Pro**, a more *realistic* and more *comprehensive* bilingual long-context benchmark containing 1,500 samples.

<div align="center">
  <img src="images/pipeline.png" width="100%"/>
</div>


## Dataset

LongBench Pro is available in the [HF Dataset](), and the specific data details can be found in the [Arxiv paper]().

<div align="center">
  <img src="images/bench_comparison.png" width="100%"/>
</div>


## Inference-Settings

All models are run **three times** using their **default parameters (with the temperature set to 1.0 when no default is provided)**. According to model characteristics, thinking models and mixed-thinking models use non-thinking prompts; the former reports thinking scores, while the latter reports scores with thinking mode enabled and disabled, respectively. Non-thinking (instruct) models report the corresponding scores based on the prompt type. For generation settings, **the output length is set to 1k for non-thinking scores and 8k for thinking scores (32k for models that support a 256k context length)**. When the input is excessively long, a middle-truncation strategy is applied.

## Main-Results

<div align="center">
  <img src="images/main_result.png" width="100%"/>
</div>

**We derive the following key insights (refer to our paper for a more detailed analysis):**

- Training optimization outperforms mere size scaling.

- The claimed context length is decoupled from the effective context length.

- "Thinking" ability mitigates linguistic capability bias.

- Extreme tasks reveal substantial capability gaps.

- "Native thinking" serves as a key breakthrough.

- Mixed-thinking models achieve Pareto optimality.

## Evaluation

### 1. Code, Dataset, and Model Downloads

Run the following command to download the evaluation code.

```bash
git clone https://github.com/caskcsg/longcontext.git
cd longcontext/LongBench_Pro
```

Run the following code to download the evaluation dataset.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "caskcsg/LongBench_Pro",
    repo_type="dataset",
    local_dir="./dataset",
    local_dir_use_symlinks=False
)
```

Run the following command to download the model for partial evaluation.

```bash
huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir ./model/Qwen3-Embedding-8B
```

**The directory structure is as follows:**

```
LongBench_Pro
├── dataset
│   └── longbench_pro.json
├── model
│   ├── Qwen3-Embedding-8B
│   └── Tokenizers
│       └── qwen
├── modules
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── model_manager.py
│   └── utils.py
├── README.md
└── requirements.txt
```

### 2. Requirements

Run the following script to install the dependencies.

```bash
python3 -m venv longbench_pro
source longbench_pro/bin/activate
pip install -r requirements.txt
```

### 3. Deployment of the model under evaluation

You need to deploy the model under evaluation so that it supports the **OpenAI-compatible API**. We recommend using [**vLLM**](https://github.com/vllm-project/vllm) or [**SGLang**](https://github.com/sgl-project/sglang).
.

### 4. Inference and Evaluation

All supported parameters can be found in ```main.py```. Since inference may fail, we recommend performing inference and evaluation separately. We take the *thinking* evaluation of *Qwen3-235B-A22B-Instruct-2507* as an example, assuming that the model’s inference URL after deployment is *http://127.0.0.1:8000/v1*.

**Inference**
```python
python main.py \
    --model_name "Qwen3-235B-A22B-Instruct-2507" \
    --tokenizer_path "model/Tokenizers/qwen" \
    --context_max_length 224000 \
    --url "http://127.0.0.1:8000/v1" \
    --temperature 0.7 \
    --max_new_tokens 32768 \
    --bon_num 3 \
    --thinking_enabled \
    --only_infer
```

**Evaluation**
```python
python main.py \
    --model_name "Qwen3-235B-A22B-Instruct-2507" \
    --context_max_length 224000 \
    --bon_num 3 \
    --thinking_enabled \
    --only_eval
```

<!-- ## Visualization
 -->

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email ZiYang (`chenziyang@iie.ac.cn`) and XingWu (`wuxing@iie.ac.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use LongBench-Pro in your work:

*Coming Soon...*