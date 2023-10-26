# CYCLE
Artifacts for the paper: "CYCLE: Learning to Iteratively Program by Self-Refining the Faulty Code"

## Install Dependency

### Step-1: Setup Conda virtenv

```sh
conda create -n cycle Python=3.8;
conda activate cycle;
pip install requirements.txt;
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
### Step-2: Configure your [Accelerate](https://huggingface.co/docs/accelerate/index)

Choose your configuration by running `accelerate config`. Our default config can be referred at default_config.yaml

## Download Data and Pre-trained Checkpoints

Our data and pre-trained checkpoints are available at: 
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10022023.svg)](https://doi.org/10.5281/zenodo.10022023)

## How to Use

### Quick Demo

```sh
bash demo.sh
```
### Featured scripts

- Phase-I Fine-tuning & Phase-2 Self-refine Training: [train_causal_lm.py](./train_causal_lm.py)
- Code Generation: [nl2code_inference.py](./nl2code_inference.py)
- Exectution Framework for Evaluation: [exec_eval/](./exec_eval/)