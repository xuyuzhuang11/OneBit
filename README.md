# OneBit: Towards Extremely Low-bit Large Language Models

<div align="center">
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

Please refer to our [📃paper](https://arxiv.org/abs/2402.11295) for more details.

## News

- [2024/05/20] We released the training code, evaluation code, and data for reproducing our result.

- [2024/02/17] We released the preprint paper to share this OneBit discovery.

# 🏷️ Benchmarks

- LLaMA-7B

| Method | Wiki2 | C4 | Winograde | Hellaswag | PIQA | BoolQ | ARC-e | ARC-c | avg. |
|--------|-------|----|-----------|-----------|------|-------|-------|-------|------|
| FP16 | 5.68 | 7.08 | 66.85 | 72.99 | 77.37 | 73.21 | 52.53 | 41.38 | 64.06 |
| OmniQuant | 15.34 | 26.21 | 52.96 | 43.68 | 62.79 | 58.69 | 41.54 | 29.35 | 48.17 |
| OneBit | 10.20 | 11.41 | 58.80 | 51.60 | 67.57 | 56.94 | 42.60 | 30.20 | 51.29 |

- LLaMA-13B
