"""
C2: Age-Matched Retraining — Train AST from scratch on participants aged 60-80 only.
This eliminates age confounding at the training stage, not just post-hoc.

Key difference from primary experiment:
- Only participants aged 60-80 included (removes the 27.5-year age gap)
- Same AST architecture, hyperparameters, and 5-fold CV protocol
- Compares against: post-hoc age-matched AUC (0.787), age-only baseline
"""

import sys
import json
import time
import copy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import ASTModel, ASTConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.linear_model import LogisticRegression

# ── Paths ──
ROOT = Path('/data0/b2ai-voice/3.0.0')
SPEC = ROOT / 'features' / 'torchaudio_mel_spectrogram.parquet'
PD_PHEN = ROOT / 'phenotype' / 'diagnosis' / 'parkinsons_disease.tsv'
CTRL_PHEN = ROOT / 'phenotype' / 'diagnosis' / 'control.tsv'
DEMO_PATH = ROOT / 'phenotype' / 'demographics' / 'demographics.tsv'
RESULTS_DIR = Path('/home/saptpurk/bridge2ai-voice-parkinsons-ast/results/v3')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AGE_MIN, AGE_MAX = 60, 80
TARGET_SEQ_LEN = 1024
N_FOLDS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── 1. Load spectrograms ──
print('Loading spectrograms...')
pf = pq.ParquetFile(SPEC)
parts = []
for i in range(pf.num_row_groups):
    parts.append(pf.read_row_group(i, columns=['participant_id','session_id','task_name','mel_spectrogram','n_frames']).to_pandas())
spec = pd.concat(parts, ignore_index=True)
spec['participant_id'] = spec['participant_id'].astype(str).str.zfill(6)
print(f'Total recordings: {len(spec)}')

# ── 2. Build PD labels ──
pd_df = pd.read_csv(PD_PHEN, sep='\t')
ctrl_df = pd.read_csv(CTRL_PHEN, sep='\t')
pd_ids = set(pd_df['participant_id'].astype(str).str.zfill(6))
ctrl_ids = set(ctrl_df['participant_id'].astype(str).str.zfill(6))
overlap = pd_ids & ctrl_ids
ctrl_ids_clean = ctrl_ids - overlap

spec['label'] = np.nan
spec.loc[spec['participant_id'].isin(pd_ids), 'label'] = 1
spec.loc[spec['participant_id'].isin(ctrl_ids_clean), 'label'] = 0
data = spec.dropna(subset=['label']).copy()
data['label'] = data['label'].astype(int)

# ── 3. Task selection ──
SELECTED_TASKS = [
    'prolonged-vowel', 'glides-high-to-low', 'glides-low-to-high',
    'diadochokinesis-pataka', 'rainbow-passage', 'picture-description',
    'story-recall', 'maximum-phonation-time-1',
]
MIN_TIME_FRAMES = 100

data_selected = data[
    (data['task_name'].isin(SELECTED_TASKS)) &
    (data['n_frames'] >= MIN_TIME_FRAMES)
].copy()

# ── 4. Age-match: restrict to 60-80 ──
demo = pd.read_csv(DEMO_PATH, sep='\t')
demo['participant_id'] = demo['participant_id'].astype(str).str.zfill(6)

# Merge age
data_selected = data_selected.merge(demo[['participant_id', 'age', 'sex_at_birth']], on='participant_id', how='left')
data_selected = data_selected.dropna(subset=['age'])
# Handle non-numeric ages like "90 and above"
data_selected['age'] = pd.to_numeric(data_selected['age'], errors='coerce')
data_selected = data_selected.dropna(subset=['age'])
data_selected['age'] = data_selected['age'].astype(int)

# Filter by age range
data_am = data_selected[(data_selected['age'] >= AGE_MIN) & (data_selected['age'] <= AGE_MAX)].copy()

print(f'\n=== Age-Matched Cohort [{AGE_MIN}-{AGE_MAX}] ===')
print(f'Recordings: {len(data_am)}')
n_pd = data_am[data_am['label']==1]['participant_id'].nunique()
n_ctrl = data_am[data_am['label']==0]['participant_id'].nunique()
print(f'Participants: {data_am["participant_id"].nunique()} (PD: {n_pd}, Control: {n_ctrl})')
print(f'Mean age PD: {data_am[data_am["label"]==1].groupby("participant_id")["age"].first().mean():.1f}')
print(f'Mean age Ctrl: {data_am[data_am["label"]==0].groupby("participant_id")["age"].first().mean():.1f}')

# ── 5. Process spectrograms ──
def process_spectrogram(spec_raw, target_len=1024):
    spec_arr = np.stack(spec_raw).astype(np.float32)
    n_mels, time_len = spec_arr.shape
    if time_len < target_len:
        spec_arr = np.pad(spec_arr, ((0, 0), (0, target_len - time_len)), mode='reflect')
    elif time_len > target_len:
        start = (time_len - target_len) // 2
        spec_arr = spec_arr[:, start:start + target_len]
    return spec_arr

print('Processing spectrograms...')
X_list = []
for _, row in tqdm(data_am.iterrows(), total=len(data_am), desc='Processing'):
    X_list.append(process_spectrogram(row['mel_spectrogram'], TARGET_SEQ_LEN))

X_raw = np.stack(X_list)
y_raw = data_am['label'].values
participants_raw = data_am['participant_id'].values
ages_raw = data_am['age'].values
print(f'Shape: {X_raw.shape}')

# ── 6. Model definition ──
def resize_spectrogram(spec_arr, target_mel=128, target_time=1024):
    mel_ratio = target_mel / spec_arr.shape[0]
    time_ratio = target_time / spec_arr.shape[1]
    return zoom(spec_arr, (mel_ratio, time_ratio), order=1).astype(np.float32)

class ASTClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_base=False):
        super().__init__()
        if pretrained:
            self.ast = ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
            hidden_size = self.ast.config.hidden_size
        else:
            config = ASTConfig(hidden_size=768, num_hidden_layers=12,
                             num_attention_heads=12, intermediate_size=3072,
                             max_length=1024, num_mel_bins=128)
            self.ast = ASTModel(config)
            hidden_size = 768
        if freeze_base:
            for param in self.ast.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        outputs = self.ast(input_values=x)
        return self.classifier(outputs.pooler_output)

class ASTDataset(Dataset):
    def __init__(self, X, y, participants, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.participants = np.array(participants)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            if np.random.random() < 0.5:
                t = np.random.randint(50, 150)
                t0 = np.random.randint(0, max(1, x.shape[1] - t))
                x[:, t0:t0+t] = 0
            if np.random.random() < 0.5:
                f = np.random.randint(10, 30)
                f0 = np.random.randint(0, max(1, x.shape[0] - f))
                x[f0:f0+f, :] = 0
        return {'inputs': x, 'labels': self.y[idx], 'participant': self.participants[idx]}

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

def evaluate_fold(model, loader, device):
    model.eval()
    all_probs, all_labels, all_parts = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['inputs'].to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch['labels'].numpy())
            all_parts.extend(batch['participant'])
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_parts = np.array(all_parts)
    unique_parts = np.unique(all_parts)
    part_probs, part_labels = [], []
    for p in unique_parts:
        mask = all_parts == p
        part_probs.append(all_probs[mask].mean())
        part_labels.append(all_labels[mask][0])
    return np.array(part_probs), np.array(part_labels), unique_parts

# ── 7. 5-Fold Cross-Validation on age-matched cohort ──
unique_participants = np.unique(participants_raw)
participant_labels = np.array([y_raw[participants_raw == p][0] for p in unique_participants])
participant_ages = np.array([ages_raw[participants_raw == p][0] for p in unique_participants])

print(f'\n=== Starting 5-Fold CV (age-matched) ===')
print(f'Participants: {len(unique_participants)} (PD: {participant_labels.sum():.0f}, Ctrl: {(participant_labels==0).sum():.0f})')

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_results = []
all_oof_probs = np.zeros(len(unique_participants), dtype=np.float32)
all_oof_labels = participant_labels.astype(np.int64).copy()

total_start = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(unique_participants, participant_labels)):
    print(f'\n--- Fold {fold+1}/{N_FOLDS} ---')
    train_parts = unique_participants[train_idx]
    val_parts = unique_participants[val_idx]

    train_mask = np.isin(participants_raw, train_parts)
    val_mask = np.isin(participants_raw, val_parts)

    X_train = X_raw[train_mask]
    y_train = y_raw[train_mask]
    p_train = participants_raw[train_mask]

    X_val = X_raw[val_mask]
    y_val = y_raw[val_mask]
    p_val = participants_raw[val_mask]

    print(f'Train: {len(X_train)} recs from {len(train_parts)} ppl')
    print(f'Val: {len(X_val)} recs from {len(val_parts)} ppl')

    # Resize
    X_train_ast = np.stack([resize_spectrogram(x) for x in tqdm(X_train, desc='resize train', leave=False)])
    X_val_ast = np.stack([resize_spectrogram(x) for x in tqdm(X_val, desc='resize val', leave=False)])

    # Fold-specific normalization
    fold_mean = X_train_ast.mean()
    fold_std = X_train_ast.std()
    X_train_ast = (X_train_ast - fold_mean) / (fold_std + 1e-8)
    X_val_ast = (X_val_ast - fold_mean) / (fold_std + 1e-8)

    # Datasets
    train_ds = ASTDataset(X_train_ast, y_train, p_train, augment=True)
    val_ds = ASTDataset(X_val_ast, y_val, p_val, augment=False)

    # Balanced sampler
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Fresh model
    model = ASTClassifier(num_classes=2, pretrained=True, freeze_base=False).to(device)
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    head_params = [p for n, p in model.named_parameters() if 'classifier' in n]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-6, 'weight_decay': 0.01},
        {'params': head_params, 'lr': 5e-4, 'weight_decay': 0.01}
    ], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)

    cc = np.bincount(y_train)
    cw = (cc.sum() / (2.0 * cc)).astype(np.float32)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    best_score = 0
    best_state = None
    patience_counter = 0

    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        part_probs, part_labels_f, _ = evaluate_fold(model, val_loader, device)
        if len(np.unique(part_labels_f)) > 1:
            auc = roc_auc_score(part_labels_f, part_probs)
            fpr, tpr, thresholds = roc_curve(part_labels_f, part_probs)
            opt_idx = np.argmax(tpr - fpr)
            preds = (part_probs >= thresholds[opt_idx]).astype(int)
            f1 = f1_score(part_labels_f, preds, zero_division=0)
        else:
            auc, f1 = 0.5, 0.0

        score = 0.4 * auc + 0.6 * f1
        if score > best_score + 0.01:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = '<-- best'
        else:
            patience_counter += 1
            marker = ''

        print(f'  Ep {epoch+1:02d} | loss {total_loss/len(train_loader):.4f} | AUC {auc:.3f} | F1 {f1:.3f} {marker}')
        if patience_counter >= 10:
            print(f'  Early stop at epoch {epoch+1}')
            break

    model.load_state_dict(best_state)
    part_probs, part_labels_f, val_pids = evaluate_fold(model, val_loader, device)

    # Save fold model
    torch.save(model.state_dict(), str(RESULTS_DIR / f'ast_pd_v3_agematched_fold{fold+1}.pt'))

    # Store OOF
    for i, pid in enumerate(val_pids):
        idx_oof = np.where(unique_participants == pid)[0][0]
        all_oof_probs[idx_oof] = part_probs[i]

    fold_auc = roc_auc_score(part_labels_f, part_probs)
    fpr, tpr, thresholds = roc_curve(part_labels_f, part_probs)
    opt_idx = np.argmax(tpr - fpr)
    preds = (part_probs >= thresholds[opt_idx]).astype(int)

    fold_results.append({
        'fold': fold + 1,
        'auc': float(fold_auc),
        'f1': float(f1_score(part_labels_f, preds, zero_division=0)),
        'recall': float(recall_score(part_labels_f, preds, zero_division=0)),
        'precision': float(precision_score(part_labels_f, preds, zero_division=0)),
    })
    print(f'Fold {fold+1}: AUC={fold_auc:.4f}')

    del model, optimizer, train_ds, val_ds
    torch.cuda.empty_cache()

total_time = time.time() - total_start
print(f'\nTotal CV time: {total_time/60:.1f} min')

# ── 8. OOF Summary ──
aucs = [r['auc'] for r in fold_results]
oof_auc = roc_auc_score(all_oof_labels, all_oof_probs)
fpr, tpr, thresholds = roc_curve(all_oof_labels, all_oof_probs)
opt_idx = np.argmax(tpr - fpr)
oof_preds = (all_oof_probs >= thresholds[opt_idx]).astype(int)
oof_f1 = f1_score(all_oof_labels, oof_preds, zero_division=0)
oof_rec = recall_score(all_oof_labels, oof_preds, zero_division=0)
oof_prec = precision_score(all_oof_labels, oof_preds, zero_division=0)

n = N_FOLDS
t_crit = stats.t.ppf(0.975, df=n-1)
auc_mean = np.mean(aucs)
auc_sd = np.std(aucs, ddof=1)
auc_ci = (auc_mean - t_crit*auc_sd/np.sqrt(n), auc_mean + t_crit*auc_sd/np.sqrt(n))

print(f'\n=== Age-Matched Retraining Results [{AGE_MIN}-{AGE_MAX}] ===')
print(f'Participants: {len(unique_participants)} (PD: {int(participant_labels.sum())}, Ctrl: {int((participant_labels==0).sum())})')
print(f'Mean age PD: {participant_ages[participant_labels==1].mean():.1f} ({participant_ages[participant_labels==1].std():.1f})')
print(f'Mean age Ctrl: {participant_ages[participant_labels==0].mean():.1f} ({participant_ages[participant_labels==0].std():.1f})')
print(f'Mean fold AUC: {auc_mean:.4f} (SD {auc_sd:.4f}) [95% CI: {auc_ci[0]:.4f}, {auc_ci[1]:.4f}]')
print(f'OOF AUC: {oof_auc:.4f}')
print(f'OOF F1: {oof_f1:.4f}  Recall: {oof_rec:.4f}  Precision: {oof_prec:.4f}')
for r in fold_results:
    print(f'  Fold {r["fold"]}: AUC={r["auc"]:.4f}  F1={r["f1"]:.4f}')

# ── 9. Age-only and metadata baselines on the same cohort ──
meta_features = np.column_stack([participant_ages, (participant_ages * 0)])  # age only
meta_labels = participant_labels

# Age-only LR
age_oof = np.zeros(len(meta_labels))
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(skf2.split(unique_participants, meta_labels)):
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(participant_ages[tr].reshape(-1, 1), meta_labels[tr])
    age_oof[va] = lr.predict_proba(participant_ages[va].reshape(-1, 1))[:, 1]

age_only_auc = roc_auc_score(meta_labels, age_oof)
print(f'\nAge-only LR AUC (same cohort): {age_only_auc:.4f}')

# ── 10. Comparison summary ──
print(f'\n=== COMPARISON ===')
print(f'Post-hoc age-matched AST (from full model): AUC 0.787')
print(f'Retrained age-matched AST:                  AUC {oof_auc:.3f}')
print(f'Age-only LR (same cohort):                  AUC {age_only_auc:.3f}')
print(f'Full-cohort AST:                            AUC 0.843')
print(f'Full-cohort age-only:                       AUC 0.875')

# ── 11. Save results ──
results = {
    'experiment': 'age_matched_retraining',
    'age_range': [AGE_MIN, AGE_MAX],
    'n_participants': int(len(unique_participants)),
    'n_pd': int(participant_labels.sum()),
    'n_ctrl': int((participant_labels == 0).sum()),
    'mean_age_pd': float(participant_ages[participant_labels==1].mean()),
    'mean_age_ctrl': float(participant_ages[participant_labels==0].mean()),
    'oof_auc': float(oof_auc),
    'oof_f1': float(oof_f1),
    'oof_recall': float(oof_rec),
    'oof_precision': float(oof_prec),
    'mean_fold_auc': float(auc_mean),
    'sd_fold_auc': float(auc_sd),
    'ci_95': [float(auc_ci[0]), float(auc_ci[1])],
    'fold_results': fold_results,
    'age_only_auc': float(age_only_auc),
    'posthoc_age_matched_auc': 0.787,
    'full_cohort_auc': 0.843,
    'total_time_minutes': float(total_time / 60),
}

with open(str(RESULTS_DIR / 'ast_pd_v3_agematched_retraining.json'), 'w') as f:
    json.dump(results, f, indent=2)

np.savez(str(RESULTS_DIR / 'ast_pd_v3_agematched_retraining.npz'),
    oof_probs=all_oof_probs,
    oof_labels=all_oof_labels,
    participant_ids=unique_participants,
    participant_ages=participant_ages,
)

print(f'\nSaved: {RESULTS_DIR}/ast_pd_v3_agematched_retraining.json')
print(f'Saved: {RESULTS_DIR}/ast_pd_v3_agematched_retraining.npz')
print('DONE.')
