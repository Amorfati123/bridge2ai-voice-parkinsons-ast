#!/usr/bin/env python3
"""N2: SpecAugment Ablation — Fine-tuned AST WITHOUT augmentation, 5-fold CV."""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"

import torch, random, numpy as np, copy, time, json
torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42); random.seed(42)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel, ASTConfig
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from scipy import stats as sp_stats
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

# ---- Data Loading ----
ROOT = Path('/data0/b2ai-voice/2.0.0')
pheno = pd.read_csv(ROOT / 'phenotype.tsv', sep='\t')
pf = pq.ParquetFile(ROOT / 'spectrogram.parquet')
parts = []
for i in range(pf.num_row_groups):
    parts.append(pf.read_row_group(i, columns=['participant_id','session_id','task_name','spectrogram']).to_pandas())
spec = pd.concat(parts, ignore_index=True)
print(f"pheno: {pheno.shape}, spec: {spec.shape}")

pheno['parkinsons_label'] = pheno['parkinsons'].map({'Checked':1, 'Unchecked':0})
labels = pheno[['participant_id','parkinsons_label']].dropna()
labels['parkinsons_label'] = labels['parkinsons_label'].astype(int)
data = spec.merge(labels, on='participant_id', how='inner')
data['time_frames'] = data['spectrogram'].apply(lambda s: np.stack(s).shape[1])

high_pd_tasks = [
    'Cinderella-Story', 'Productive-Vocabulary-1', 'Productive-Vocabulary-2',
    'Productive-Vocabulary-3', 'Productive-Vocabulary-4', 'Productive-Vocabulary-5',
    'Productive-Vocabulary-6', 'Word-color-Stroop', 'Random-Item-Generation',
]
data_sel = data[(data['task_name'].isin(high_pd_tasks)) & (data['time_frames'] >= 100)].copy()

def process_spectrogram_raw(spec_raw, target_len=1024):
    spec = np.stack(spec_raw).astype(np.float32)
    n_mels, time_len = spec.shape
    if time_len < target_len:
        spec = np.pad(spec, ((0, 0), (0, target_len - time_len)), mode='reflect')
    elif time_len > target_len:
        start = (time_len - target_len) // 2
        spec = spec[:, start:start + target_len]
    return spec

X_raw = np.stack([process_spectrogram_raw(row['spectrogram'], 1024) for _, row in tqdm(data_sel.iterrows(), total=len(data_sel), desc="Load specs")])
y_raw = data_sel['parkinsons_label'].values
participants_raw = data_sel['participant_id'].values

unique_participants = np.unique(participants_raw)
participant_labels = np.array([y_raw[participants_raw == p][0] for p in unique_participants])
print(f"Data: {X_raw.shape}, {len(unique_participants)} participants, {participant_labels.sum()} PD+")

# ---- AST Model (same as PD notebook) ----
class ASTClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_base=False):
        super().__init__()
        if pretrained:
            self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            hidden_size = self.ast.config.hidden_size
        else:
            config = ASTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                               intermediate_size=3072, max_length=1024, num_mel_bins=128)
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

class ASTDatasetNoAug(torch.utils.data.Dataset):
    """Dataset with NO augmentation (the key difference for this ablation)."""
    def __init__(self, X, y, participants):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.participants = np.array(participants)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return {'inputs': self.X[idx], 'labels': self.y[idx], 'participant': self.participants[idx]}

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

def resize_spec(spec, target_mel=128, target_time=1024):
    return zoom(spec, (target_mel / spec.shape[0], target_time / spec.shape[1]), order=1).astype(np.float32)

def evaluate_fold(model, loader, device):
    model.eval()
    all_probs, all_labels, all_parts = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['inputs'].to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs); all_labels.extend(batch['labels'].numpy()); all_parts.extend(batch['participant'])
    all_probs, all_labels, all_parts = np.array(all_probs), np.array(all_labels), np.array(all_parts)
    unique_parts = np.unique(all_parts)
    part_probs = np.array([all_probs[all_parts == p].mean() for p in unique_parts])
    part_labels = np.array([all_labels[all_parts == p][0] for p in unique_parts])
    return part_probs, part_labels, unique_parts

# ---- 5-Fold CV (NO AUGMENTATION) ----
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = []
oof_probs = np.zeros(len(unique_participants), dtype=np.float32)
oof_labels = participant_labels.astype(np.int64).copy()

total_start = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(unique_participants, participant_labels)):
    print(f"\n{'='*60}")
    print(f"Fold {fold+1}/{N_FOLDS} (Fine-tuned AST — NO SpecAugment)")
    print(f"{'='*60}")

    train_parts = unique_participants[train_idx]
    val_parts = unique_participants[val_idx]
    train_mask = np.isin(participants_raw, train_parts)
    val_mask = np.isin(participants_raw, val_parts)

    print(f"Train: {train_mask.sum()} recordings from {len(train_parts)} participants")
    print(f"Val:   {val_mask.sum()} recordings from {len(val_parts)} participants")

    # Resize spectrograms to 128x1024
    X_train = np.stack([resize_spec(x) for x in tqdm(X_raw[train_mask], desc="resize train", leave=False)])
    X_val = np.stack([resize_spec(x) for x in tqdm(X_raw[val_mask], desc="resize val", leave=False)])

    fold_mean, fold_std = X_train.mean(), X_train.std()
    X_train = (X_train - fold_mean) / (fold_std + 1e-8)
    X_val = (X_val - fold_mean) / (fold_std + 1e-8)

    # KEY DIFFERENCE: augment=False for BOTH train and val
    train_ds = ASTDatasetNoAug(X_train, y_raw[train_mask], participants_raw[train_mask])
    val_ds = ASTDatasetNoAug(X_val, y_raw[val_mask], participants_raw[val_mask])

    # Balanced sampler
    cc = np.bincount(y_raw[train_mask])
    sample_weights = (1.0 / cc)[y_raw[train_mask]]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Fresh model (FULL fine-tuning, same as original PD notebook)
    model = ASTClassifier(num_classes=2, pretrained=True, freeze_base=False).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Differential LR (same as original)
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-6},
        {'params': head_params, 'lr': 5e-4},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)

    cw = (cc.sum() / (2.0 * cc)).astype(np.float32)
    criterion = FocalLoss(alpha=torch.tensor(cw, dtype=torch.float32).to(device), gamma=2.0)
    print(f"Class weights: {cw}")

    best_score, best_state, patience_counter = 0, None, 0

    for epoch in range(30):
        model.train()
        for batch in train_loader:
            inputs, labels_b = batch['inputs'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        part_probs_e, part_labels_e, _ = evaluate_fold(model, val_loader, device)
        if len(np.unique(part_labels_e)) > 1:
            auc = roc_auc_score(part_labels_e, part_probs_e)
            fpr, tpr, thresholds = roc_curve(part_labels_e, part_probs_e)
            opt_thresh = thresholds[np.argmax(tpr - fpr)]
            f1_opt = f1_score(part_labels_e, (part_probs_e >= opt_thresh).astype(int), zero_division=0)
        else:
            auc, f1_opt = 0.5, 0.0

        score = 0.4 * auc + 0.6 * f1_opt
        if score > best_score + 0.01:
            best_score, best_state, patience_counter = score, copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    part_probs_f, part_labels_f, val_pids = evaluate_fold(model, val_loader, device)

    for i, pid in enumerate(val_pids):
        oof_probs[np.where(unique_participants == pid)[0][0]] = part_probs_f[i]

    fold_auc = roc_auc_score(part_labels_f, part_probs_f)
    fpr, tpr, thresholds = roc_curve(part_labels_f, part_probs_f)
    opt_thresh = thresholds[np.argmax(tpr - fpr)]
    preds_opt = (part_probs_f >= opt_thresh).astype(int)

    fold_results.append({
        'fold': fold + 1,
        'auc': float(fold_auc),
        'f1': float(f1_score(part_labels_f, preds_opt, zero_division=0)),
        'recall': float(recall_score(part_labels_f, preds_opt, zero_division=0)),
        'precision': float(precision_score(part_labels_f, preds_opt, zero_division=0)),
    })
    print(f"Fold {fold+1}: AUC={fold_auc:.4f}, F1={fold_results[-1]['f1']:.4f}, "
          f"Rec={fold_results[-1]['recall']:.4f}, Prec={fold_results[-1]['precision']:.4f}")

    del model, optimizer
    torch.cuda.empty_cache()

total_time = time.time() - total_start

# ---- Summary ----
print("\n" + "="*60)
print("FINE-TUNED AST (NO SpecAugment) - Per-fold results")
print("="*60)
for r in fold_results:
    print(f"  Fold {r['fold']}: AUC={r['auc']:.4f}  F1={r['f1']:.4f}  Rec={r['recall']:.4f}  Prec={r['precision']:.4f}")

n = N_FOLDS
t_crit = sp_stats.t.ppf(0.975, df=n-1)
for m_name in ['auc', 'f1', 'recall', 'precision']:
    vals = [r[m_name] for r in fold_results]
    m, sd = np.mean(vals), np.std(vals, ddof=1)
    ci_lo, ci_hi = m - t_crit * sd / np.sqrt(n), m + t_crit * sd / np.sqrt(n)
    print(f"  Mean {m_name.upper()}: {m:.4f} +/- {sd:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

oof_auc = roc_auc_score(oof_labels, oof_probs)
fpr, tpr, thresholds = roc_curve(oof_labels, oof_probs)
oof_thresh = thresholds[np.argmax(tpr - fpr)]
oof_preds = (oof_probs >= oof_thresh).astype(int)

print(f"\nOOF AUC:       {oof_auc:.4f}")
print(f"OOF F1:        {f1_score(oof_labels, oof_preds, zero_division=0):.4f} (threshold={oof_thresh:.3f})")
print(f"OOF Recall:    {recall_score(oof_labels, oof_preds, zero_division=0):.4f}")
print(f"OOF Precision: {precision_score(oof_labels, oof_preds, zero_division=0):.4f}")
print(f"\nTotal CV time: {total_time/60:.1f} minutes")

# Compare with original (with SpecAugment)
try:
    ft_results = np.load("ast_pd_selected_cv_results.npz", allow_pickle=True)
    ft_oof_auc = roc_auc_score(ft_results['oof_labels'], ft_results['oof_probs'])
    delta = oof_auc - ft_oof_auc
    print(f"\n--- SPECAUGMENT ABLATION COMPARISON ---")
    print(f"  With SpecAugment (original):    OOF AUC = {ft_oof_auc:.4f}")
    print(f"  Without SpecAugment (this run):  OOF AUC = {oof_auc:.4f}")
    print(f"  Difference:                      {delta:+.4f} AUC")
    if delta < -0.01:
        print(f"  -> SpecAugment improves performance by {abs(delta):.3f} AUC")
    elif delta > 0.01:
        print(f"  -> SpecAugment hurts performance by {delta:.3f} AUC (unexpected)")
    else:
        print(f"  -> SpecAugment has minimal effect ({delta:+.3f} AUC)")
except Exception as e:
    print(f"Could not load original results for comparison: {e}")

np.savez("specaugment_ablation_cv_results.npz",
    oof_probs=oof_probs, oof_labels=oof_labels, participant_ids=unique_participants,
    fold_aucs=np.array([r['auc'] for r in fold_results]),
    fold_f1s=np.array([r['f1'] for r in fold_results]),
    fold_recalls=np.array([r['recall'] for r in fold_results]),
    fold_precisions=np.array([r['precision'] for r in fold_results]),
)
print("Saved: specaugment_ablation_cv_results.npz")

summary = {
    'model': 'Fine-tuned AST (NO SpecAugment)',
    'oof_auc': float(oof_auc),
    'oof_f1': float(f1_score(oof_labels, oof_preds, zero_division=0)),
    'oof_recall': float(recall_score(oof_labels, oof_preds, zero_division=0)),
    'oof_precision': float(precision_score(oof_labels, oof_preds, zero_division=0)),
    'fold_results': fold_results,
    'total_time_minutes': round(total_time / 60, 1),
}
with open("specaugment_ablation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: specaugment_ablation_summary.json")
