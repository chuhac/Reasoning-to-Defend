# Reasoning-to-Defend

<a href="https://arxiv.org/abs/2502.12970">
<img src="https://img.shields.io/badge/arXiv-2502.12970-red.svg">
</a>
<a href="https://huggingface.co/datasets/chuhac/R2D-R1">
  <img src="https://img.shields.io/badge/HF-dataset-blue.svg">
</a>
<a href="https://opensource.org/licenses/MIT">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
</a>



Code for paper

> [***Reasoning-to-Defend:* Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking**](https://arxiv.org/abs/2502.12970)   
> *Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha* 

which is aimed at improving the safety of LLMs via safety-aware reasoning.

![overview](assets/overview.png)

## News:
* Aug 20th, 2025: R2D is accepted by EMNLP 2025 Main Conference.
* May 31st, 2025: Dataset released, and is available at [huggingface](https://huggingface.co/datasets/chuhac/R2D-R1).
* Mar 9th, 2025: Code for R2D has been released. The dataset is currently being organized.
* Feb 18th, 2025: R2D's paper is publicly available at [arxiv](https://arxiv.org/abs/2502.12970).


## Usage

### Training *R2D* Model

In order to better conduct Contrastive Pivot Optimization proposed in the paper, we expand the vocabularies before training.
```shell
cd r2d_train
bash expand_and_train.sh
```

### Evaluation

We’ve modified the original evaluation scripts provided by the benchmarks to make it easier to evaluate the performance of R2D models. These scripts can be found in the respective folders.


## Acknowledgement
* llm-attacks: <a href="https://github.com/llm-attacks/llm-attacks">https://github.com/llm-attacks/llm-attacks</a>
* HarmBench: <a href="https://github.com/centerforaisafety/HarmBench">https://github.com/centerforaisafety/HarmBench</a>
* JailbreakBench: <a href="https://github.com/JailbreakBench/jailbreakbench">https://github.com/JailbreakBench/jailbreakbench</a>
* XSTest: <a href="https://github.com/paul-rottger/xstest">https://github.com/paul-rottger/xstest</a>
* Transformers: <a href="https://github.com/huggingface/transformers">https://github.com/huggingface/transformers</a>
* DeepSpeed: <a href="https://github.com/microsoft/DeepSpeed">https://github.com/microsoft/DeepSpeed</a>
* accelerate: <a href="https://github.com/huggingface/accelerate">https://github.com/huggingface/accelerate</a>
* vLLM: <a href="https://github.com/vllm-project/vllm">https://github.com/vllm-project/vllm</a>

## Citation
If you find this repository useful, please cite our paper:
```
@article{zhu2025reasoning,
  title={Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking},
  author={Zhu, Junda and Yan, Lingyong and Wang, Shuaiqiang and Yin, Dawei and Sha, Lei},
  journal={arXiv preprint arXiv:2502.12970},
  year={2025}
}
```
