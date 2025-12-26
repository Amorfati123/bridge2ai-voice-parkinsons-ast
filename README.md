# bridge2ai-voice-parkinsons-ast

This repository contains experimental notebooks for a voice-based Parkinson’s disease detection study using the **Bridge2AI Voice dataset** and a **pretrained Audio Spectrogram Transformer (AST)**.

The study investigates whether structured voice tasks encode reproducible acoustic signatures of Parkinson’s disease and evaluates transformer-based models under strict **participant-level validation** to avoid data leakage. Multiple cohort and modeling strategies are explored to understand the impact of task selection, class imbalance, and participant metadata on model performance.

## Notebooks

### 1. AST_selected_tasks_Spectrograms_PD.ipynb
Primary workflow using a curated subset of structured tasks with higher Parkinson’s disease prevalence. This notebook:
- Applies fixed-length spectrogram preprocessing
- Fine-tunes a pretrained Audio Spectrogram Transformer
- Reports **5-fold participant-level cross-validation** results

This notebook represents the main reference experiment.

---

### 2. Metadata_AST_selected_tasks_Spectrograms_PD.ipynb
Extension of the primary workflow incorporating **participant-level metadata (age and sex)** using a late-fusion (soft voting) strategy. This notebook evaluates whether demographic information provides complementary predictive signal beyond voice alone.

---

### 3. AllTasks_AST_selected_tasks_Spectrograms_PD.ipynb
Ablation experiment using **all available voice tasks** with a minimum recording-length constraint. This notebook evaluates whether AST performance generalizes across heterogeneous task types under substantial class imbalance.

---

## Key Methodological Features

- Participant-level train/test splits and cross-validation
- Fixed-length spectrogram preprocessing with padding and cropping
- Fine-tuning of a pretrained Audio Spectrogram Transformer
- Balanced sampling and focal loss for class imbalance
- Late-fusion evaluation of voice predictions and metadata
- Explicit comparison of cohort selection versus modeling-stage imbalance handling

