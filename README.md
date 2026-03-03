# 🎵 BioFoundation: Foundation Models for Bioacoustics

<div align="center">

## A Comparative Review of Foundation Models for Bioacoustics 🤗

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch_Lightning-792ee5?logo=pytorch-lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

*A comprehensive evaluation framework for foundation models in bioacoustic analysis*

</div>

---

## 🔍 Overview

This repository contains the official implementation and evaluation framework for our paper **"Foundation Models for Bioacoustics: A Comparative Review"**. We present a systematic comparison of state-of-the-art foundation models across multiple bioacoustic benchmarks, providing insights into their effectiveness for animal sound classification and analysis.

### 🎯 Key Features

- **Comprehensive Evaluation**: Systematic comparison of 12+ foundation models
- **Multiple Benchmarks**: Evaluation on BEANS and BirdSet datasets
- **Flexible Framework**: Easy-to-use scripts for reproducing experiments
- **Standardized Protocols**: Linear probing, attentive probing, and fine-tuning evaluations
- **Rich Documentation**: Detailed configuration and setup instructions

### 📊 Supported Models

Our framework evaluates the following foundation models:

**Baseline General Audio Models:**
- **AudioMAE** 
- **BEATs** 
- **EAT** 

**Bioacoustic Foundation Models:**
- **AVES** 
- **BEATs NLM** 
- **BioLingual** 
- **Bird AVES**
- **BirdMAE**
- **ConvNeXt_BS** 
- **Perch**
- **ProtoCLR** 
- **SurfPerch**
- **ViT INS**

Some of the models need local Checkpoint files, which can be downloaded using the following links. Make sure to adjust the path in the config file based on your file location:

| Model | GitHub | Checkpoint | Config |
| :--- | :--- | :--- | :--- |
| **BEATs** | [beats](https://github.com/microsoft/unilm/tree/master/beats) | [BEATs_iter3_plus_AS2M.pt](https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf) | - |
| **AVES** | [aves](https://github.com/earthspecies/aves?tab=readme-ov-file#pretrained-models) | [aves-base-bio.torchaudio.pt](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt) | [aves-base-bio.torchaudio.model_config.json](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.model_config.json) |
| **Bird AVES** | [aves](https://github.com/earthspecies/aves?tab=readme-ov-file#pretrained-models) | [birdaves-biox-base.torchaudio.pt](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.torchaudio.pt) | [birdaves-biox-base.torchaudio.model_config.json](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.torchaudio.model_config.json) |
| **BEATs NLM** | [NatureLM-audio](https://github.com/earthspecies/NatureLM-audio) | Extract? | - |
| **BirdMAE** | [Bird-MAE](https://github.com/DBD-research-group/Bird-MAE) | ? HF? | - |
| **ProtoCLR** | [ProtoCLR](https://github.com/ilyassmoummad/ProtoCLR) | [protoclr.pth](https://huggingface.co/ilyassmoummad/ProtoCLR/resolve/main/protoclr.pth) | - |
| **ViT INS** | [iNatSounds](https://github.com/cvl-umass/iNatSounds) | [vit_single_mixup.pt](https://drive.google.com/file/d/1dr1bLURsXiPcX8xQOzs1eO5y2htullM7/view?usp=share_link) | - |


### 🗂️ Datasets

- **BEANS**: Benchmark of Animal Sounds
  - Watkins Marine Mammal Dataset (31 classes)
  - Bat Calls (10 classes) 
  - CBI Bird Dataset (264 classes)
  - Dog Barks (10 classes)
  - HumBugDB Mosquito Dataset (14 classes)

- **BirdSet**: Comprehensive bird sound benchmark
  - 8 datasets: PER, POW, NES, UHH, HSN, NBP, SSW, SNE


---

## 🚀 Quick Start

### Installation

#### Using Devcontainer (Recommended)

We provide a preconfigured development container for easy setup:

```bash
git submodule update --init --recursive
```

#### Manual Installation

Install dependencies using [Poetry](https://python-poetry.org/):

```bash
poetry install
poetry shell
```

### 🧪 Running Experiments

#### BirdSet Experiments

Use our convenient `run_birdset.sh` script to evaluate models on BirdSet datasets:

```bash
# Run all models on all BirdSet datasets
./projects/biofoundation/scripts/run_birdset.sh

# Run specific models
./projects/biofoundation/scripts/run_birdset.sh --models perch,aves,audiomae

# Run on specific datasets
./projects/biofoundation/scripts/run_birdset.sh --datasets PER,POW,NES

# Custom configuration
./projects/biofoundation/scripts/run_birdset.sh --models perch --datasets PER --seeds 1,2,3 --gpu 0
```

#### BEANS Experiments

Use our `run_beans.sh` script for BEANS benchmark evaluation:

```bash
# Run all models on all BEANS datasets
./projects/biofoundation/scripts/run_beans.sh

# Run specific models
./projects/biofoundation/scripts/run_beans.sh --models perch,aves

# Run on specific datasets
./projects/biofoundation/scripts/run_beans.sh --datasets beans_watkins,beans_cbi

# Custom configuration
./projects/biofoundation/scripts/run_beans.sh --models perch --datasets beans_watkins --seeds 1,2,3 --gpu 0
```

#### Manual Experiment Execution

For more granular control, you can run individual experiments:

```bash
# BirdSet linear probing
./projects/biofoundation/train.sh experiment=birdset/linearprobing/{model_name}

# BEANS linear probing  
./projects/biofoundation/train.sh experiment=beans/linearprobing/{model_name}
```
---

## 📊 Results and Analysis

### Generating Results Tables

We provide automated table generation for our comprehensive results analysis:

```bash
# Download results data from WandB report
# https://wandb.ai/deepbirddetect/BioFoundation/reports/Latex-Table-Data--VmlldzoxMjEyODQ0Ng

# Generate LaTeX tables
python projects/biofoundation/results/latex/new_table.py
```

The script requires `beans.csv` and `birdset.csv` files in the same directory, which can be downloaded from our [WandB Report](https://wandb.ai/deepbirddetect/BioFoundation/reports/Latex-Table-Data--VmlldzoxMjEyODQ0Ng).

### Hyperparameter Optimization with WandB Sweeps

We use Weights & Biases Sweeps for systematic hyperparameter optimization:

```bash
# Start a sweep
wandb sweep sweeps/base_grid.yaml

# Run sweep agents
wandb agent <sweep_id>

# Multi-GPU sweep execution
projects/biofoundation/sweeps/sweep.sh <gpu_id> <sweep_id>
```

Available sweep configurations:
- `sweeps/base_grid.yaml`: Grid search for basic parameters
- `sweeps/classifier.yaml`: Bayesian optimization for classifier architectures

---

## 📝 Configuration

### BEANS Dataset Configuration

To run experiments on specific BEANS datasets, modify the experiment configuration:

```yaml
datamodule:
  dataset:
    dataset_name: beans_watkins # Choose dataset
    hf_path: DBD-research-group/beans_watkins # HuggingFace path
    hf_name: default
    n_classes: 31 # Number of classes
```

**Available BEANS Datasets:**

|Dataset|Classes|Description|
|-------|-------|-----------|
|`beans_watkins`|31|Marine mammal vocalizations|
|`beans_bats`|10|Bat echolocation calls|
|`beans_cbi`|264|Cornell Bird Identification|
|`beans_dogs`|10|Dog bark classifications|
|`beans_humbugdb`|14|Mosquito wing-beat sounds|




