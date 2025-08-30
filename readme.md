# Distributed SimCLR for 3D Medical MRI (SSL3D Challenge Extension by Team PVG)

## Overview

This repository extends the official nnSSL OpenMind Challenge codebase with improvements for scalable distributed training and enhanced 3D augmentations.

We build on the challenge baseline by adding multi-node distributed training with torchrun (tested on three or more nodes with four H100 GPUs each), the Adam optimizer combined with a cosine annealing learning rate schedule, large-batch training with batch size 120 and 275 iterations per epoch, and improved MONAI-based 3D augmentations designed for medical MRI.

This implementation was used in our participation in the challenge, scaling beyond single-node training while maintaining reproducibility with the provided framework.

## Installation
Please see the original repo (https://github.com/MIC-DKFZ/nnssl/tree/openneuro) for complete installation instructions. We summarize these steps briefly:

1. Clone this repository (based on the challenge starter code).  
```bash
git clone <your-env>
cd <your-repo>
pip install -e .
```
2. Set up required environment variables (same as original repo):
  ```bash
  export nnssl_raw=<raw_dataset_path>
  export nnssl_preprocessed=<preprocessed_dataset_path>
  export nnssl_results=<results_path>
```

3. Activate your environment.
  You may either follow the challenge creators’ environment setup or load your own conda/virtualenv:

## Distributed Training

Training is performed using SLURM together with torchrun to enable multi-node distributed training. In our setup, we trained on three nodes with four GPUs each, for a total of twelve GPUs. SLURM handles job scheduling across nodes, while torchrun coordinates communication by assigning each node a rank and connecting them through the master address and port.

## Improvements Over Baseline

The following modifications were introduced compared to the challenge baseline:

- The optimizer and scheduler were replaced with Adam and cosine annealing, improving stability and convergence.  
- The batch size was increased to 120, with 275 iterations per epoch to balance updates and efficiency.  
- MONAI augmentations were integrated for 3D medical imaging, including stronger spatial and intensity-based transformations to improve robustness, and resizing to the downstream task input size (160x160x160).

## Workflow

The training workflow follows the same structure as the challenge framework:

- Raw data preparation requires creation of a pretrain_data.json file describing dataset contents.  
- Preprocessing fingerprints, plans, normalizes, resamples, and prepares data for training.  
- Pretraining launches the updated SimCLR trainer with the distributed training setup.  
- Adaptation fine-tunes checkpoints for downstream segmentation or classification using the linked frameworks.  

The original documentation provides detailed guidance on dataset formats, preprocessing steps, and additional utilities.

## References

- Original repo: nnSSL Challenge Codebase (https://github.com/MIC-DKFZ/nnssl/tree/openneuro)
- Paper: *An OpenMind for 3D medical vision self-supervised learning* (https://arxiv.org/pdf/2412.17041v2)
- Augmentations: MONAI Transforms (https://monai.io)

## License

This repository inherits the license of the original nnSSL challenge code. Please ensure compliance with its terms.
