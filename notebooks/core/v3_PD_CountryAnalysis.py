"""
C4: Country-Level Analysis — Assess performance by country (USA vs Canada)
as a proxy for multi-site analysis.

Bridge2AI v3.0.0 doesn't expose explicit site identifiers, but country
(USA=709, Canada=202) provides a coarse proxy for geographic/institutional variation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

RESULTS_DIR = Path('/home/saptpurk/bridge2ai-voice-parkinsons-ast/results/v3')
ROOT = Path('/data0/b2ai-voice/3.0.0')
DEMO_PATH = ROOT / 'phenotype' / 'demographics' / 'demographics.tsv'

# ── 1. Load existing OOF predictions from the primary PD experiment ──
cv = np.load(str(RESULTS_DIR / 'ast_pd_v3_cv_results.npz'), allow_pickle=True)
oof_probs = cv['oof_probs']
oof_labels = cv['oof_labels']
participant_ids = cv['participant_ids']

print(f'Loaded OOF predictions for {len(participant_ids)} participants')

# ── 2. Load demographics ──
demo = pd.read_csv(DEMO_PATH, sep='\t')
demo['participant_id'] = demo['participant_id'].astype(str).str.zfill(6)
# Handle non-numeric ages
demo['age'] = pd.to_numeric(demo['age'], errors='coerce')

# Build merged dataframe
df = pd.DataFrame({
    'participant_id': participant_ids,
    'oof_prob': oof_probs,
    'label': oof_labels,
})
# Use left join on the OOF predictions (253 participants)
df = df.merge(demo[['participant_id', 'age', 'sex_at_birth', 'country']].drop_duplicates('participant_id'),
              on='participant_id', how='left')

print(f'Total participants: {len(df)}')
print(f'Matched with demographics: {df["country"].notna().sum()}')
print(f'Country distribution:')
print(df['country'].value_counts())

# ── 3. Per-country analysis ──
results = {}
for country in df['country'].dropna().unique():
    sub = df[df['country'] == country]
    n = len(sub)
    n_pd = int(sub['label'].sum())
    n_ctrl = int((sub['label'] == 0).sum())

    if n_pd < 5 or n_ctrl < 5:
        print(f'\n{country}: Skipping (too few: {n_pd} PD, {n_ctrl} Ctrl)')
        continue

    auc = roc_auc_score(sub['label'], sub['oof_prob'])
    fpr, tpr, thresholds = roc_curve(sub['label'], sub['oof_prob'])
    opt_idx = np.argmax(tpr - fpr)
    preds = (sub['oof_prob'].values >= thresholds[opt_idx]).astype(int)
    f1 = f1_score(sub['label'], preds, zero_division=0)

    mean_age_pd = sub[sub['label']==1]['age'].mean()
    mean_age_ctrl = sub[sub['label']==0]['age'].mean()
    age_gap = mean_age_pd - mean_age_ctrl

    results[country] = {
        'n': int(n),
        'n_pd': n_pd,
        'n_ctrl': n_ctrl,
        'auc': float(auc),
        'f1': float(f1),
        'mean_age_pd': float(mean_age_pd),
        'mean_age_ctrl': float(mean_age_ctrl),
        'age_gap': float(age_gap),
        'pct_male_pd': float((sub[sub['label']==1]['sex_at_birth']=='Male').mean()),
        'pct_male_ctrl': float((sub[sub['label']==0]['sex_at_birth']=='Male').mean()),
    }

    print(f'\n{country}: N={n} (PD={n_pd}, Ctrl={n_ctrl})')
    print(f'  AUC: {auc:.4f}  F1: {f1:.4f}')
    print(f'  Mean age PD: {mean_age_pd:.1f}, Ctrl: {mean_age_ctrl:.1f} (gap: {age_gap:.1f})')

# ── 4. Age-matched country analysis ──
print(f'\n=== Age-Matched Country Analysis [60-80] ===')
df_am = df[(df['age'] >= 60) & (df['age'] <= 80)]

for country in df_am['country'].dropna().unique():
    sub = df_am[df_am['country'] == country]
    n_pd = int(sub['label'].sum())
    n_ctrl = int((sub['label'] == 0).sum())

    if n_pd < 3 or n_ctrl < 3:
        print(f'{country}: Skipping (too few for age-matched: {n_pd} PD, {n_ctrl} Ctrl)')
        continue

    auc = roc_auc_score(sub['label'], sub['oof_prob'])
    mean_age_pd = sub[sub['label']==1]['age'].mean()
    mean_age_ctrl = sub[sub['label']==0]['age'].mean()

    results[f'{country}_agematched_60_80'] = {
        'n': int(len(sub)),
        'n_pd': n_pd,
        'n_ctrl': n_ctrl,
        'auc': float(auc),
        'mean_age_pd': float(mean_age_pd),
        'mean_age_ctrl': float(mean_age_ctrl),
        'age_gap': float(mean_age_pd - mean_age_ctrl),
    }

    print(f'{country} [60-80]: N={len(sub)} (PD={n_pd}, Ctrl={n_ctrl}) AUC={auc:.4f}')
    print(f'  Age PD: {mean_age_pd:.1f}, Ctrl: {mean_age_ctrl:.1f}')

# ── 5. Save ──
output = {
    'experiment': 'country_level_analysis',
    'note': 'Country (USA/Canada) used as proxy for multi-site analysis. No explicit site IDs in v3.0.0.',
    'results': results,
}

with open(str(RESULTS_DIR / 'country_analysis.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nSaved: {RESULTS_DIR}/country_analysis.json')
print('DONE.')
