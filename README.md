# Efficient LLM and Multimodal Foundation Model Survey

This repo contains the paper list and figures for [A Survey of Backpropagation-free Training For LLMs](./main-survey-fwd.pdf).

## Abstract

Large language models (LLMs) have achieved remarkable performance in various downstreaming tasks. 
However, the training of LLMs is computationally expensive and requires a large amount of memory. 
To address this issue, backpropagation-free (BP-free) training  has been proposed as a promising approach to reduce the computational and memory costs of training LLMs. 
In this survey, we provide a comprehensive overview of BP-free training for LLMs from the perspective of mainstream BP-free training methods and their optimizations for LLMs.
The goal of this survey is to provide a comprehensive understanding of BP-free training for LLMs and to inspire future research in this area.

<!-- ## Citation

```
@article{xu2024a,
    title = {A Survey of Resource-efficient LLM and Multimodal Foundation Models},
    author = {Xu, Mengwei and Yin, Wangsong and Cai, Dongqi and Yi, Rongjie
    and Xu, Daliang and Wang, Qipeng and Wu, Bingyang and Zhao, Yihao and Yang, Chen
    and Wang, Shihe and Zhang, Qiyang and Lu, Zhenyan and Zhang, Li and Wang, Shangguang
    and Li, Yuanchun, and Liu Yunxin and Jin, Xin and Liu, Xuanzhe},
    journal={arXiv preprint arXiv:2401.08092},
    year = {2024}
}
``` -->

## Contribute

If we leave out any important papers, please let us know in the Issues and we will include them in the next version.

We will actively maintain the survey and the Github repo.

## Table of Contents

- [BP-free Methods](#bp-free-methods)
    - [Perturbated Model](#perturbated-model)
        - [Forward Gradient](#forward-gradient)
        - [Zeroth-order Optimization](#zeroth-order-optimization)
        - [Evolution Strategy](#evolution-strategy)
    - [Perturbated Input](#perturbated-input)
    - [No Perturbation](#no-perturbation)
- [BP-Free LLM](#bp-free-llm)
    - [Parameter-Efficient Tuning](#parameter-efficient-tuning)
    - [Full-Parameter Tuning](#full-parameter-tuning)


## BP-free Methods

### Perturbated Model

- Gradients without Backpropagation. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2202.08587)

- Can Forward Gradient Match  Backpropagation? *[ICLR'23]* [[Paper]](https://arxiv.org/abs/2306.06968) 

- Scaling Forward Gradient With Local Losses. *[ICLR'23]* [[Paper]](https://arxiv.org/abs/2210.03310) [[Code]](https://github.com/google-research/google-research/tree/master/local_forward_gradient)

#### Forward Gradient

#### Zeroth-order Optimization

#### Evolution Strategy

### Perturbated Input

### No Perturbation

## BP-free LLM

### Parameter-Efficient Tuning

### Full-Parameter Tuning
