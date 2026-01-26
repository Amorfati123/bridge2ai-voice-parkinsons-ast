# bridge2ai-voice-parkinsons-ast

This repository contains experimental notebooks for voice-based disease detection studies using the **Bridge2AI Voice dataset** and a **pretrained Audio Spectrogram Transformer (AST)**.

The work investigates whether structured voice tasks encode reproducible acoustic signatures of neurological, psychiatric, and cognitive conditions. All experiments enforce **strict participant-level validation** to prevent data leakage and to reflect realistic clinical generalization. Multiple cohort selection and modeling strategies are explored to understand the effects of task choice, class imbalance, and evaluation thresholds on downstream performance.

## Notebooks

### 1. AST_selected_tasks_Spectrograms_PD.ipynb
Primary workflow for Parkinson’s disease detection using a curated subset of structured speech tasks with higher disease prevalence. This notebook:
- Applies fixed-length spectrogram preprocessing
- Fine-tunes a pretrained Audio Spectrogram Transformer
- Reports **5-fold participant-level cross-validation** results

This notebook serves as the main reference experiment.

---

### 2. Metadata_AST_selected_tasks_Spectrograms_PD.ipynb
Extension of the Parkinson’s disease workflow incorporating **participant-level metadata (age and sex)** via late fusion (soft voting). This notebook evaluates whether demographic variables provide complementary signal beyond voice alone.

---

### 3. AllTasks_AST_selected_tasks_Spectrograms_PD.ipynb
Ablation experiment using **all available voice tasks** with a minimum recording-length constraint. This notebook evaluates whether AST performance generalizes across heterogeneous task types under substantial class imbalance.

---

### 4. depression_AST_selected_tasks_Spectrograms.ipynb
Voice-based **depression detection** workflow following the same modeling and validation protocol as the Parkinson’s experiments.  
Key features:
- Selection of **high-prevalence, language-heavy tasks** (e.g., Animal Fluency, Open Response Questions)
- Balanced participant cohort with near-equal positive/negative samples
- **5-fold participant-level cross-validation** with out-of-fold evaluation
- Analysis of threshold sensitivity and operating-point tradeoffs

This experiment demonstrates stronger and more stable performance than respiratory conditions, suggesting that **cognitive–linguistic tasks encode clearer depressive vocal signatures**.

---

### 5. dementia_AST_selected_tasks_Spectrograms.ipynb
Voice-based **dementia (Alzheimer’s / MCI)** detection workflow using the same AST pipeline.  
Key features:
- Task selection emphasizing **memory, lexical retrieval, and executive function** (e.g., Productive Vocabulary, Random Item Generation, Cinderella Story)
- Moderate class imbalance handled via balanced sampling and focal loss
- **Participant-level 5-fold cross-validation** with both per-fold and out-of-fold evaluation
- Analysis of threshold variability and clinical operating points

This notebook shows the strongest overall performance, consistent with the hypothesis that **cognitive impairment produces robust and task-consistent vocal changes**.

---

## Key Methodological Features

- Strict participant-level train/test splits and cross-validation
- Fixed-length spectrogram preprocessing with padding and cropping
- Fine-tuning of a pretrained Audio Spectrogram Transformer (AST)
- Balanced sampling and focal loss to address class imbalance
- Participant-level aggregation of predictions across tasks
- Explicit analysis of threshold selection, operating points, and out-of-fold performance
- Comparative evaluation across neurological, psychiatric, and cognitive conditions
