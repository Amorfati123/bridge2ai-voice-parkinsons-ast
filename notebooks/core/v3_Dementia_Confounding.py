"""
Dementia Confounding Analysis — Assess age confounding in dementia screening.

Mirrors the PD confounding analysis: compares AST OOF predictions against
age-only and metadata-only baselines, evaluates age-matched subgroups.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path('/home/saptpurk/bridge2ai-voice-parkinsons-ast/results/v3')
ROOT = Path('/data0/b2ai-voice/3.0.0')
DEMO_PATH = ROOT / 'phenotype' / 'demographics' / 'demographics.tsv'

# ── 1. Load existing OOF predictions from the dementia experiment ──
cv = np.load(str(RESULTS_DIR / 'ast_dementia_v3_cv_results.npz'), allow_pickle=True)
oof_probs = cv['oof_probs']
oof_labels = cv['oof_labels']
participant_ids = cv['participant_ids']

print(f'Loaded OOF predictions for {len(participant_ids)} participants')
print(f'Dementia: {(oof_labels==1).sum()}, Control: {(oof_labels==0).sum()}')
print(f'AST OOF AUC: {roc_auc_score(oof_labels, oof_probs):.4f}')

# ── 2. Load demographics ──
demo = pd.read_csv(DEMO_PATH, sep='\t')
demo['participant_id'] = demo['participant_id'].astype(str).str.zfill(6)
demo['age'] = pd.to_numeric(demo['age'], errors='coerce')

df = pd.DataFrame({
    'participant_id': participant_ids,
    'oof_prob': oof_probs,
    'label': oof_labels,
})
df = df.merge(
    demo[['participant_id', 'age', 'sex_at_birth']].drop_duplicates('participant_id'),
    on='participant_id', how='left'
)

n_with_demo = df['age'].notna().sum()
print(f'Participants with demographics: {n_with_demo} / {len(df)}')

# Demographics summary
dem_ages = df[df['label']==1]['age']
ctrl_ages = df[df['label']==0]['age']
print(f'\nDementia: mean age {dem_ages.mean():.1f} (SD {dem_ages.std():.1f}), range [{dem_ages.min():.0f}, {dem_ages.max():.0f}]')
print(f'Control:  mean age {ctrl_ages.mean():.1f} (SD {ctrl_ages.std():.1f}), range [{ctrl_ages.min():.0f}, {ctrl_ages.max():.0f}]')
print(f'Age gap: {dem_ages.mean() - ctrl_ages.mean():.1f} years')
print(f'Dementia male%: {(df[df["label"]==1]["sex_at_birth"]=="Male").mean()*100:.1f}%')
print(f'Control male%:  {(df[df["label"]==0]["sex_at_birth"]=="Male").mean()*100:.1f}%')

# ── 3. Metadata-only classifiers (same CV folds) ──
df_meta = df.dropna(subset=['age']).copy()
meta_participants = df_meta['participant_id'].values
meta_labels = df_meta['label'].values.astype(int)

# Encode sex
df_meta['sex_num'] = (df_meta['sex_at_birth'] == 'Male').astype(int)

results = {}

# --- Age-only ---
age_features = df_meta[['age']].values
age_oof = np.zeros(len(df_meta), dtype=np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(meta_participants, meta_labels)):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(age_features[train_idx])
    X_val = scaler.transform(age_features[val_idx])
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train, meta_labels[train_idx])
    age_oof[val_idx] = lr.predict_proba(X_val)[:, 1]

age_auc = roc_auc_score(meta_labels, age_oof)
print(f'\nAge-only AUC: {age_auc:.4f}')

# --- Metadata (age + sex) ---
meta_features = df_meta[['age', 'sex_num']].values
meta_oof = np.zeros(len(df_meta), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(skf.split(meta_participants, meta_labels)):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(meta_features[train_idx])
    X_val = scaler.transform(meta_features[val_idx])
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train, meta_labels[train_idx])
    meta_oof[val_idx] = lr.predict_proba(X_val)[:, 1]

meta_auc = roc_auc_score(meta_labels, meta_oof)
print(f'Metadata (age+sex) AUC: {meta_auc:.4f}')

# --- Sex-only ---
sex_features = df_meta[['sex_num']].values
sex_oof = np.zeros(len(df_meta), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(skf.split(meta_participants, meta_labels)):
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(sex_features[train_idx], meta_labels[train_idx])
    sex_oof[val_idx] = lr.predict_proba(sex_features[val_idx])[:, 1]

sex_auc = roc_auc_score(meta_labels, sex_oof)
print(f'Sex-only AUC: {sex_auc:.4f}')

# AST AUC on the subset with demographics
ast_sub = df_meta['oof_prob'].values
ast_sub_auc = roc_auc_score(meta_labels, ast_sub)
print(f'AST AUC (demo subset): {ast_sub_auc:.4f}')

results['full_cohort'] = {
    'n': int(len(df_meta)),
    'n_dementia': int(meta_labels.sum()),
    'n_ctrl': int((meta_labels == 0).sum()),
    'mean_age_dementia': float(dem_ages.mean()),
    'mean_age_ctrl': float(ctrl_ages.mean()),
    'age_gap': float(dem_ages.mean() - ctrl_ages.mean()),
    'ast_auc': float(ast_sub_auc),
    'age_only_auc': float(age_auc),
    'metadata_auc': float(meta_auc),
    'sex_only_auc': float(sex_auc),
}

# ── 4. Age-matched subgroup analysis ──
print('\n=== Age-Matched Subgroup Analysis ===')

for label, age_lo, age_hi in [('55_85', 55, 85), ('60_80', 60, 80), ('65_85', 65, 85)]:
    sub = df_meta[(df_meta['age'] >= age_lo) & (df_meta['age'] <= age_hi)]
    n_dem = int(sub['label'].sum())
    n_ctrl = int((sub['label'] == 0).sum())

    if n_dem < 5 or n_ctrl < 5:
        print(f'[{age_lo}-{age_hi}]: Skipping (too few: {n_dem} dementia, {n_ctrl} ctrl)')
        continue

    sub_labels = sub['label'].values.astype(int)

    # Get positional indices into df_meta for the subgroup
    pos_indices = [df_meta.index.get_loc(i) for i in sub.index]

    # AST
    ast_auc_sub = roc_auc_score(sub_labels, sub['oof_prob'].values)

    # Age-only
    age_auc_sub = roc_auc_score(sub_labels, age_oof[pos_indices])

    # Metadata
    meta_auc_sub = roc_auc_score(sub_labels, meta_oof[pos_indices])

    mean_age_dem = sub[sub['label']==1]['age'].mean()
    mean_age_ctrl = sub[sub['label']==0]['age'].mean()
    age_gap = mean_age_dem - mean_age_ctrl

    print(f'\n[{age_lo}-{age_hi}]: N={len(sub)} (Dementia={n_dem}, Ctrl={n_ctrl})')
    print(f'  Mean age: Dementia={mean_age_dem:.1f}, Ctrl={mean_age_ctrl:.1f} (gap={age_gap:.1f})')
    print(f'  AST AUC: {ast_auc_sub:.4f}')
    print(f'  Age-only AUC: {age_auc_sub:.4f}')
    print(f'  Metadata AUC: {meta_auc_sub:.4f}')

    results[f'agematched_{label}'] = {
        'n': int(len(sub)),
        'n_dementia': n_dem,
        'n_ctrl': n_ctrl,
        'mean_age_dementia': float(mean_age_dem),
        'mean_age_ctrl': float(mean_age_ctrl),
        'age_gap': float(age_gap),
        'ast_auc': float(ast_auc_sub),
        'age_only_auc': float(age_auc_sub),
        'metadata_auc': float(meta_auc_sub),
    }

# ── 5. Country analysis for dementia ──
print('\n=== Country Analysis (Dementia) ===')
df_country = df.merge(
    demo[['participant_id', 'country']].drop_duplicates('participant_id'),
    on='participant_id', how='left'
)
print(df_country['country'].value_counts())

for country in df_country['country'].dropna().unique():
    sub = df_country[df_country['country'] == country]
    n_dem = int(sub['label'].sum())
    n_ctrl = int((sub['label'] == 0).sum())
    print(f'{country}: N={len(sub)} (Dementia={n_dem}, Ctrl={n_ctrl})')
    if n_dem >= 5 and n_ctrl >= 5:
        auc = roc_auc_score(sub['label'], sub['oof_prob'])
        print(f'  AST AUC: {auc:.4f}')
        results[f'country_{country}'] = {
            'n': int(len(sub)),
            'n_dementia': n_dem,
            'n_ctrl': n_ctrl,
            'auc': float(auc),
        }

# ── 6. Save ──
output = {
    'experiment': 'dementia_confounding_analysis',
    'note': 'Age confounding analysis for dementia screening, mirroring PD analysis.',
    'results': results,
}

with open(str(RESULTS_DIR / 'dementia_confounding.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nSaved: {RESULTS_DIR}/dementia_confounding.json')
print('DONE.')
