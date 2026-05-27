# Visual Waveform Analytics & Multi-Task Deep Learning Framework

An advanced computer vision and signals engineering framework designed for automated web-scale data harvesting, stratified synthetic data generation, multi-task network optimization, real-time edge screening, and deep mathematical parameter regression. This repository leverages customized **ResNet-18 Deep Convolutional Neural Network (CNN)** architectures to simultaneously classify signal types and extract precise continuous structural equations directly from raw image matrices.

---

##  System Architecture & Codebase Map

The pipeline cleanly decouples data generation, dataset maintenance, multi-task neural network optimization, and interactive deployment runmodes across your repository folders:

```text
waveform-detection-model/
├── simulations/                  # Data Generation & Synthesis Strategy Modules
│   ├── gen.py                    # Strategy A: Stratified Cartesian quadrant signal synthesis
│   └── generator.py              # Strategy B: Micro-wobble humanized line simulator
│
├── src/                          # Core Code, Utilities, & Live Inference Runmodes
│   ├── noone.py                  # Web automation & raw chart asset crawler
│   ├── augmenter.py              # Image-space transformation & dataset scaling utility
│   ├── split.py                  # Reproducible data frame train/val partitioning
│   ├── fileswithc.py             # Directory reconciliation & collision-free class-merger
│   ├── run.py                    # Mode 1: Live OpenCV webcam binary screening stream
│   ├── run2.py                   # Mode 2: Graphic vector parameter regression tester
│   └── test.py                   # Mode 3: 11-Class continuous function characterizer
│
├── training/                     # Deep Learning Network Training Engines
│   ├── trainerup.py              # Custom PyTorch pure continuous coordinate regression loop
│   └── L4_trainer.py             # Multi-Task Learning (MTL) transfer network optimization
│
├── dataset_raw/                  # Raw web-scraped assets (Local only / git-ignored)
├── humanized_linear_dataset/      # Hand-drawn simulated datasets (Local only / git-ignored)
├── humanized_linear_dataset1/     # Quadrant-stratified datasets (Local only / git-ignored)
└── .gitignore                    # Excludes heavy datasets and local paths (<1 MB repo footprint)