# Efficient LLM and Multimodal Foundation Model Survey

This repo contains the paper list for [A Survey of Backpropagation-free Training For LLMs](./main-survey-fwd.pdf).

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

- Gradients without Backpropagation. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2202.08587)

- Learning by Directional Gradient Descent. *[ICLR'22]* [[Paper]](https://openreview.net/forum?id=5i7lJLuhTm)

- Optimization without Backpropagation. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2209.06302)

- Scaling Forward Gradient With Local Losses. *[ICLR'23]* [[Paper]](https://arxiv.org/abs/2210.03310) [[Code]](https://github.com/google-research/google-research/tree/master/local_forward_gradient)

- Can Forward Gradient Match  Backpropagation? *[ICLR'23]* [[Paper]](https://arxiv.org/abs/2306.06968) 

- Low-variance Forward Gradients using Direct Feedback Alignment and momentum. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2212.07282)

- How to Guess a Gradient. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2312.04709)

#### Zeroth-order Optimization

- Does Federated Learning Really Need Backpropagation? *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2301.12195) [[Code]](https://github.com/FengHZ/BAFFLE)

- Fine-Tuning Language Models with Just Forward Passes. *[NeurIPS'23]* [[Paper]](https://arxiv.org/abs/2305.17333) [[Code]](https://github.com/princeton-nlp/MeZO)

- DPZero: Dimension-Independent and Differentially Private Zeroth-Order Optimization. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2310.09639)

- DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training. *[ICLR'24]* [[Paper]](https://arxiv.org/abs/2310.02025) [[Code]](https://github.com/OPTML-Group/DeepZero)


#### Evolution Strategy

- Black-Box Tuning for Language-Model-as-a-Service. *[ICML'22]* [[Paper]](https://arxiv.org/abs/2201.03514) [[Code]](https://github.com/txsun1997/Black-Box-Tuning)

- BBTv2: Towards a Gradient-Free Future with Large Language Models. *[EMNLP'22]* [[Paper]](https://arxiv.org/abs/2205.11200) [[Code]](https://github.com/txsun1997/Black-Box-Tuning)

- Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies. *[ICML'21]* [[Paper]](https://arxiv.org/abs/2112.13835)

- Low-Variance Gradient Estimation in Unrolled Computation Graphs with ES-Single. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2304.11153)
### Perturbated Input

- The Forward-Forward Algorithm: Some Preliminary Investigations. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2212.13345)[[Code]](https://github.com/pytorch/examples/tree/main/mnist_forward_forward)
- Graph Neural Networks Go Forward-Forward. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2302.05282)
- The Predictive Forward-Forward Algorithm. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2301.01452)[[Code]](https://github.com/ago109/predictive-forward-forward)
- Contrastive-Signal-Dependent Plasticity: Forward-Forward Learning of Spiking Neural Systems. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2303.18187)
- Training Convolutional Neural Networks with the Forward-Forward Algorithm. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2312.14924)
- Backpropagation-free Training of Deep Physical Neural Networks. *[Science'23]* [[Paper]](https://www.science.org/doi/abs/10.1126/science.adi8474)
- Forward-Forward Training of an Optical Neural Network. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.19170)
- µ-FF: On-Device Forward-Forward Training Algorithm for Microcontrollers. *[SMARTCOMP'23]* [[Paper]](https://ieeexplore.ieee.org/abstract/document/10207585)
- Error-driven Input Modulation: Solving the Credit Assignment Problem without a Backward Pass. *[ICML'22]* [[Paper]](https://proceedings.mlr.press/v162/dellaferrera22a.html?trk=public_post_comment-text)[[Code]](https://github.com/GiorgiaD/PEPITA)
- Suitability of Forward-Forward and PEPITA Learning to MLCommons-Tiny benchmarks. *[COINS'23]* [[Paper]](https://ieeexplore.ieee.org/document/10189239)[[Code]](https://github.com/fabrizioaymone/suitability-of-Forward-Forwardand-PEPITA-learning)

### No Perturbation

- Neural Network Learning without Backpropagation. *[IEEE Transactions on Neural Networks'10]* [[Paper]](https://ieeexplore.ieee.org/abstract/document/5580116)
- The HSIC Bottleneck: Deep Learning without Back-Propagation. *[AAAI'20]* [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5950)[[Code]](https://github.com/choasma/HSIC-Bottleneck)
- Building Deep Random Ferns Without Backpropagation. *[IEEE Access'20]* [[Paper]](https://ieeexplore.ieee.org/abstract/document/8952691)

## BP-free LLM

### Parameter-Efficient Tuning

- Black-Box Tuning for Language-Model-as-a-Service. *[ICML'22]* [[Paper]](https://proceedings.mlr.press/v162/sun22e.html)[[Code]](https://github.com/txsun1997/Black-Box-Tuning)
- BBTv2: Towards a Gradient-Free Future with Large Language Models. *[EMNLP'22]* [[Paper]](https://aclanthology.org/2022.emnlp-main.259/)[[Code]](https://github.com/txsun1997/Black-Box-Tuning)
- Make Prompt-based Black-Box Tuning Colorful: Boosting Model Generalization from Three Orthogonal Perspectives. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.08088)[[Code]](https://github.com/QiushiSun/BBT-RGB)
- Clip-Tuning: Towards Derivative-free Prompt Learning with a Mixture of Rewards. *[EMNLP'22]* [[Paper]](https://aclanthology.org/2022.findings-emnlp.8/)
- RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning. *[EMNLP'22]* [[Paper]](https://aclanthology.org/2022.emnlp-main.222/)[[Code]](https://github.com/mingkaid/rl-prompt)
- Black-box Prompt Learning for Pre-trained Language Models. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2201.08531)[[Code]](https://github.com/shizhediao/Black-Box-Prompt-Learning)
- PromptBoosting: Black-Box Text Classification with Ten Forward Passes. *[ICML'23]* [[Paper]](https://proceedings.mlr.press/v202/hou23b.html)[[Code]](https://github.com/UCSB-NLP-Chang/PromptBoosting)
- GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models. *[arXiv'22]* [[Paper]](https://arxiv.org/abs/2203.07281)[[Code]](https://github.com/archiki/GrIPS)
- Efficient Federated Prompt Tuning for Black-box Large Pre-trained Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2310.03123)
- Iterative Forward Tuning Boosts In-context Learning in Language Models. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.13016)[[Code]](https://github.com/AlibabaResearch/DAMO-ConvAI)
- FwdLLM: Efficient FedLLM using Forward Gradient. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2308.13894)[[Code]](https://github.com/UbiquitousLearning/FwdLLM)
- HyperTuning: Toward Adapting Large Language Models without Back-propagation. *[ICML'23]* [[Paper]](https://proceedings.mlr.press/v202/phang23a.html)

### Full-Parameter Tuning

- Backpropagation Free Transformers. *[NeurIPS'20]* [[Paper]](https://dinkofranceschi.com/docs/bft.pdf)
- Forward Learning of Large Language Models by Consumer Devices. *[Electronics'24]* [[Paper]](https://www.mdpi.com/2079-9292/13/2/402)[[Code]](https://github.com/fabrizioaymone/forward-learning-of-LLMs-to-consmer-devices)
- Fine-Tuning Language Models with Just Forward Passes. *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2305.17333)[[Code]](https://github.com/princeton-nlp/MeZO)
- Federated Full-Parameter Tuning  of Billion-Sized Language Models with Communication Cost under 18 Kilobytes.  *[arXiv'23]* [[Paper]](https://arxiv.org/abs/2312.06353)[[Code]](https://github.com/alibaba/FederatedScope/tree/FedKSeed)