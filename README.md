# Audio Spectrogram Transformers for Voice-Based Disease Detection

**Task-Specific Biomarkers Across Neurological and Psychiatric Conditions**

## Overview

This repository contains the code and analysis for a study evaluating a unified pretrained Audio Spectrogram Transformer (AST) pipeline for participant-level speech-based screening across three clinically distinct conditions - **Parkinson's disease (PD)**, **dementia**, and **depression** - using the [Bridge2AI Voice Dataset (v2.0.0)](https://physionet.org/content/b2ai-voice/2.0.0/).

The same architecture, preprocessing, training procedure, and evaluation protocol are applied to all three conditions. The study demonstrates that screening performance depends critically on **speech task selection** rather than model-level modifications, positioning task design as a condition-dependent decision for clinical deployment.

All metrics are participant-level, evaluated under five-fold stratified cross-validation with strict participant-level separation. Standard deviations use `ddof=1` (sample SD).

## Repository Structure

```
├── PD_AST_selected_tasks_Spectrograms___Metadata_eval.ipynb
│       Primary PD analysis (9 selected tasks) + metadata late-fusion evaluation
│
├── PD_AllTasks_AST_Spectrograms.ipynb
│       PD sensitivity analysis using all 49 tasks (≥100-frame filter)
│
├── Dementia_AST_selected_tasks_Spectrograms.ipynb
│       Primary dementia (Alzheimer's / MCI) analysis
│
├── Depression_AST_selected_tasks_Spectrograms.ipynb
│       Primary depression analysis
│
├── requirements.txt              # Pinned package versions for reproducibility
├── LICENSE
└── README.md
```

## Notebooks

### PD Selected Tasks + Metadata Evaluation
**`PD_AST_selected_tasks_Spectrograms___Metadata_eval.ipynb`**

Primary Parkinson's disease experiment using 9 high-prevalence structured speech tasks. Includes the full pipeline (preprocessing → training → 5-fold CV → OOF evaluation → publication figures), plus a metadata late-fusion analysis evaluating whether age and sex provide complementary signal beyond voice. Also generates the task selection comparison figure (Selected vs All Tasks) and per-fold AUC strip plot.

### PD All Tasks
**`PD_AllTasks_AST_Spectrograms.ipynb`**

Sensitivity analysis using all available tasks with a ≥100-frame minimum length filter. Uses a different class weighting scheme (`[1.0, neg/pos]`), no learning rate scheduler, and a relaxed early stopping threshold (`1e-6`). These differences from the primary pipeline are documented in the manuscript.

### Dementia Selected Tasks
**`Dementia_AST_selected_tasks_Spectrograms.ipynb`**

Voice-based dementia detection using 9 tasks emphasizing memory, lexical retrieval, and executive function. Shows the strongest overall performance, consistent with the hypothesis that cognitive impairment produces robust task-consistent vocal changes.

### Depression Selected Tasks
**`Depression_AST_selected_tasks_Spectrograms.ipynb`**

Voice-based depression detection using 2 cognitive-linguistic tasks. A balanced cohort with near-equal positive/negative samples. Performance suggests depression-related vocal signatures are subtler than neurological conditions.

## Dataset

This study uses the [Bridge2AI Voice Dataset v2.0.0](https://physionet.org/content/b2ai-voice/2.0.0/), which provides log-Mel spectrograms (201 Mel bins, variable time frames) and participant-level phenotype data for 442 participants across multiple speech tasks.

**Access:** The dataset requires credentialed access through PhysioNet. Follow the instructions on the dataset page to obtain access.

## Pipeline Summary

1. **Task Selection:** Condition-specific high-prevalence subsets (9 tasks for PD/dementia, 2 for depression)
2. **Preprocessing:** Temporal standardization (reflect-pad / center-crop to 1024 frames) → frequency resize (201 → 128 Mel bins) → fold-specific z-score normalization
3. **Model:** Pretrained AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) with full fine-tuning, ~86.4 M parameters
4. **Training:** Focal loss with dynamic per-fold inverse class-frequency weights, AdamW with differential learning rates (backbone 5 × 10⁻⁶, head 5 × 10⁻⁴), cosine annealing, SpecAugment (time mask 50-150, frequency mask 10-30), early stopping on composite AUC + F1 score
5. **Evaluation:** Participant-level 5-fold stratified CV, out-of-fold (OOF) aggregation, threshold optimization via Youden's J, 95% confidence intervals with `ddof=1`

## Reproducibility

Every notebook begins with a deterministic seed cell:

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Note:** Minor numerical variation may still occur across GPU hardware and CUDA versions. The PD Selected Tasks notebook saves OOF predictions (`.npz`) and fold models (`.pt`), which downstream cells (metadata evaluation, figures) load to avoid re-training discrepancies.

## Installation

```bash
pip install -r requirements.txt
```

## Citation

```
Shukla S, Naliyatthaliyazchayil P, Gichoya J, Purkayastha S. Audio Spectrogram Transformers
for Voice-Based Disease Detection: Task-Specific Biomarkers Across Neurological and
Psychiatric Conditions. 2026.
```

## License

This project is for research purposes. See [LICENSE](LICENSE) for details. The Bridge2AI Voice Dataset is subject to its own [data use agreement](https://physionet.org/content/b2ai-voice/2.0.0/).
