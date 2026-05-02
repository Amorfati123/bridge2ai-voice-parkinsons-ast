# Demographic Confounding in Voice-Based Parkinson’s Disease Screening

This repository accompanies our study examining how demographic imbalance can inflate the performance of deep-learning models for speech-based disease detection. Using the Bridge2AI Voice Dataset v3.0.0, we showed that a simple logistic regression classifier using only participant age can outperform a state-of-the-art Audio Spectrogram Transformer (AST) in Parkinson’s disease and dementia screening. We identify large age gaps between cases and controls (≈27–30 years) and site-level diagnosis biases, and we propose an age-matched evaluation protocol to disentangle disease-specific vocal signatures from demographic confounders. A survey of prior voice-biomarker studies highlights that most do not report age distributions or perform age-matched analyses, and our cross-version experiments underscore the fragility of within-dataset performance.

## Repository structure

The code and data artefacts used to reproduce the analyses are organised as follows:

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── figures/
│   ├── attention_maps_enhanced.png
│   ├── attention_maps_pd.png
│   ├── baseline_roc_comparison.png
│   ├── calibration_curves.png
│   ├── demographic_comparison.png
│   ├── figure3_pd_performance.png
│   ├── figure4_auc_distribution.png
│   ├── figure5_sensitivity.png
│   ├── per_task_contribution.png
│   └── ppg_vs_spectrogram_roc.png
├── notebooks/
│   └── core/
│       ├── Baseline_Comparisons.ipynb
│       ├── Dementia_AST_selected_tasks_Spectrograms.ipynb
│       ├── Depression_AST_selected_tasks_Spectrograms.ipynb
│       ├── PD_AST_selected_tasks_Spectrograms+Metadata_eval.ipynb
│       ├── PD_AllTasks_AST_Spectrograms.ipynb
│       ├── v3_Baselines.ipynb
│       ├── v3_CrossVersion_Validation.ipynb
│       ├── v3_Dementia_AST_Spectrograms.ipynb
│       ├── v3_Dementia_Confounding.py
│       ├── v3_Dementia_PPG_AST.ipynb
│       ├── v3_Depression_AST_Spectrograms.ipynb
│       ├── v3_PD_AST_Spectrograms.ipynb
│       ├── v3_PD_AgeMatched_Retraining.py
│       ├── v3_PD_AllTasks_AST_Spectrograms.ipynb
│       ├── v3_PD_CountryAnalysis.py
│       ├── v3_PD_Interpretability.py
│       ├── v3_PD_Interpretability_Enhanced.py
│       ├── v3_PD_PPG_1DCNN.ipynb
│       └── v3_PD_PPG_AST.ipynb
├── scripts/
│   ├── run_N1_resnet18.py
│   └── run_N2_specaugment_ablation.py
└── results/
    ├── frozen_ast_baseline_cv_results.npz
    ├── lr_baseline_cv_results.npz
    ├── resnet18_baseline_cv_results.npz
    ├── resnet18_baseline_summary.json
    ├── specaugment_ablation_cv_results.npz
    ├── specaugment_ablation_summary.json
    └── v3/
        ├── ast_dementia_v3_cv_results.npz
        ├── ast_depression_v3_cv_results.npz
        ├── ast_pd_v3_agematched_retraining.json
        ├── ast_pd_v3_agematched_retraining.npz
        ├── ast_pd_v3_alltasks_cv_results.npz
        ├── ast_pd_v3_cv_results.npz
        ├── attention_enhanced.npz
        ├── attention_enhanced_analysis.json
        ├── attention_map_analysis.json
        ├── attention_maps.npz
        ├── country_analysis.json
        ├── cross_version_validation.json
        ├── cross_version_validation_v201.json
        ├── dementia_confounding.json
        ├── frozen_ast_v3_cv_results.npz
        ├── lr_baseline_v3_cv_results.npz
        ├── metadata_experiment_v3.npz
        ├── metadata_experiment_v3_fixed.npz
        ├── ppg_1dcnn_pd_v3_cv_results.npz
        ├── ppg_ast_dementia_v3_cv_results.npz
        ├── ppg_ast_pd_v3_cv_results.npz
        ├── resnet18_v3_cv_results.npz
        ├── specaugment_ablation_v3_cv_results.npz
        ├── v3_baseline_fold_comparison.png
        └── v3_baseline_roc_comparison.png
```

## Data access

All analysis notebooks rely on credentialed access to the Bridge2AI Voice Dataset, which can be requested from PhysioNet.

## Installation

To reproduce the experiments, install the dependencies with:

```bash
pip install -r requirements.txt
```

## Reviewer-response analyses (added May 2026)

Three additional analyses were added to address reviewer comments and are appended
to the existing `v3_PD_AST_Spectrograms.ipynb` and `v3_Dementia_AST_Spectrograms.ipynb`
notebooks (cells `#11`–`#14`). They reuse the saved fold checkpoints and OOF
predictions written by the original CV cell, so they re-run end-to-end in minutes
rather than hours.

| Cell | Analysis | Output |
|---|---|---|
| `#11` | Threshold-leakage sensitivity: leave-one-fold-out (LOFO) Youden's J + fixed=0.5 | `results/v3/lofo_and_fixed_threshold_metrics_{pd,dementia}.json` |
| `#12` | Training-partition Youden's J (per fold; loads fold checkpoints, runs inference on training partition) | `results/v3/training_threshold_metrics_{pd,dementia}.json` |
| `#13` | DeLong's test for paired AUC comparisons (Sun & Xu fast O(N log N)) — AST vs age-only vs metadata, full and age-restricted | included in `delong_and_propensity_{pd,dementia}.json` |
| `#14` | 1:1 nearest-neighbor propensity-score matching on age + sex (USA-only ages 60–80; degrades gracefully when subgroup is too small, as for dementia) | `results/v3/delong_and_propensity_{pd,dementia}.json` |

Cell `#7` is now idempotent: when all five fold `.pt` checkpoints and the
`*_cv_results.npz` exist in `results/v3/`, it loads them and skips retraining. To
re-execute either notebook end-to-end without retraining, run:

```bash
papermill notebooks/core/v3_PD_AST_Spectrograms.ipynb /tmp/v3_PD_executed.ipynb \
    --kernel b2ai-venv --log-output --no-progress-bar
papermill notebooks/core/v3_Dementia_AST_Spectrograms.ipynb /tmp/v3_Dem_executed.ipynb \
    --kernel b2ai-venv --log-output --no-progress-bar
```

Each notebook completes in ~5 minutes (cells `#1`–`#6` re-process spectrograms,
`#7` loads cached checkpoints, `#12` runs training-partition inference for 5 folds).

## Citation

Please cite our work if you use this repository in your research.

## License

This project is made available for research purposes under the terms of the `LICENSE`. The Bridge2AI Voice Dataset is subject to its own data use agreement.
