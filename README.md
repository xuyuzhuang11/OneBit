# OneBit: Towards Extremely Low-bit Large Language Models

<div align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/xuyuzhuang11/OneBit" alt="GitHub Code License">
  </a>
  <a href="https://github.com/xuyuzhuang11/OneBit/commits/main">
    <img src="https://img.shields.io/github/last-commit/xuyuzhuang11/OneBit" alt="GitHub last commit">
  </a>
  <a href="https://github.com/xuyuzhuang11/OneBit/pulls">
    <img src="https://img.shields.io/badge/PRs-welcome-blue" alt="GitHub pull request">
  </a>
  <a href="https://github.com/xuyuzhuang11/OneBit">
    <img src="https://img.shields.io/github/v/release/xuyuzhuang11/OneBit" alt="GitHub release version">
  </a>
  <br>
  <img src="imgs/logo.png" style="width: 50%">
</div>

<p align="center">
  <br>
  <strong>🔥 The First 1-bit LLM Quantization Method 🔥</strong>
  <br>
  <strong>👍🏻 The SOTA Method of 1-bit LLM Quantization (till 2024.05.20)👍🏻</strong>
  <br>
  <strong>😱 Compressing 90%+ Space yet Maintaining 80%+ Performance 😱</strong>
  <br>
</p>

<div align="center">
  <img src="imgs/star.png" style="width: 50%;">
</div>

- OneBit Framework is the first 1-bit model quantization method that compressing the fully trained model from high precision format to nearly 1-bit representation.
- OneBit Framework has a novel and efficient 1-bit model architecture for LLMs, which can improve both the time (requiring device support) and space efficiency during model inference. Moreover, this architecture is more stable during quantizing LLMs.
- We want to demonstrate that OneBit is very useful in resource constrained scenarios, such as edge-side devices.

Please refer to our 📃[paper](https://arxiv.org/abs/2402.11295) for more details.

## 📰 News

- [2024/05/20] We released the training code, evaluation code, and data for reproducing our result.

- [2024/02/17] We released the preprint paper to share this OneBit discovery.

## 🏷️ Benchmarks

Please note that due to the use of different checkpoints, the evaluation results listed here have very minor differences from the results in our paper, but this does not affect the existing conclusions.

- LLaMA-7B

| Method | Wiki2 | C4 | Winograde | Hellaswag | PIQA | BoolQ | ARC-e | ARC-c | avg. |
|--------|-------|----|-----------|-----------|------|-------|-------|-------|------|
| FP16 | 5.68 | 7.08 | 66.85 | 72.99 | 77.37 | 73.21 | 52.53 | 41.38 | 64.06 |
| OmniQuant | 15.34 | 26.21 | 52.96 | 43.68 | 62.79 | **58.69** | 41.54 | 29.35 | 48.17 |
| **OneBit** | **10.20** | **11.41** | **58.80** | **51.60** | **67.57** | 56.94 | **42.60** | **30.20** | **51.29** |

- LLaMA-13B

| Method | Wiki2 | C4 | Winograde | Hellaswag | PIQA | BoolQ | ARC-e | ARC-c | avg. |
|--------|-------|----|-----------|-----------|------|-------|-------|-------|------|
| FP16 | 5.09 | 6.61 | 70.17 | 76.24 | 79.05 | 68.47 | 59.85 | 44.54 | 66.39 |
| OmniQuant | 13.43 | 19.33 | 53.83 | 54.16 | 68.99 | 62.20 | **45.50** | 30.38 | 52.51 |
| **OneBit** | **9.19** | **10.26** | **62.12** | **56.75** | **70.35** | **63.82** | 44.57 | **31.83** | **54.91** |

- LLaMA2-7B

| Method | Wiki2 | C4 | Winograde | Hellaswag | PIQA | BoolQ | ARC-e | ARC-c | avg. |
|--------|-------|----|-----------|-----------|------|-------|-------|-------|------|
| FP16 | 5.47 | 6.97 | 67.09 | 72.94 | 76.88 | 71.10 | 53.58 | 40.61 | 63.70 |
| OmniQuant | 31.21 | 64.34 | 51.22 | 33.87 | 56.53 | 59.14 | 33.63 | 24.32 | 43.12 |
| **OneBit** | **9.76** | **11.14** | **58.17** | **52.51** | **67.79** | **63.30** | **41.67** | **29.35** | **52.13** |

- LLaMA2-13B

| Method | Wiki2 | C4 | Winograde | Hellaswag | PIQA | BoolQ | ARC-e | ARC-c | avg. |
|--------|-------|----|-----------|-----------|------|-------|-------|-------|------|
| FP16 | 4.88 | 6.47 | 69.77 | 76.62 | 79.05 | 68.99 | 57.95 | 44.20 | 66.10 |
| OmniQuant | 16.88 | 27.02 | 53.20 | 50.34 | 62.24 | 62.05 | 40.66 | 29.61 | 49.68 |
| **OneBit** | **8.77** | **10.17** | **61.33** | **56.48** | **69.97** | **64.98** | **42.85** | **33.79** | **54.90** |

All the best results are highlighted in bold.

## ⬇️ Installation

Download and install the dependencies to use these codes. You should clone this repository to your local machine, enter the code folder, and install it. We suggest the installation order to be transformers first and llama_factory after. Please note that [Pytorch](https://pytorch.org/) v2.0.0 is required to quantize the models, whereas the evaluation process does not.

```bash
git clone https://github.com/xuyuzhuang11/OneBit.git
cd OneBit/transformers
pip install -e .
cd ../llama_factory
pip install -r ./requirements.txt
```

## 🏁 Quantization

## ✅ Evaluation

## ⏩ Add New Model

## 🗞 License

## ❤️ Citation