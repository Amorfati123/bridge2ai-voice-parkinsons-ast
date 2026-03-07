"""
C3: Interpretability — Extract and visualize AST attention maps for PD classification.

Extracts attention weights from the fine-tuned AST to show which spectro-temporal
regions drive PD classification. Compares attention patterns between:
- PD cases vs controls
- Full cohort vs age-matched subgroup
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
from transformers import ASTModel, ASTConfig
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

RESULTS_DIR = Path('/home/saptpurk/bridge2ai-voice-parkinsons-ast/results/v3')
FIGURES_DIR = Path('/home/saptpurk/bridge2ai-voice-parkinsons-ast/figures')
ROOT = Path('/data0/b2ai-voice/3.0.0')
SPEC = ROOT / 'features' / 'torchaudio_mel_spectrogram.parquet'
PD_PHEN = ROOT / 'phenotype' / 'diagnosis' / 'parkinsons_disease.tsv'
CTRL_PHEN = ROOT / 'phenotype' / 'diagnosis' / 'control.tsv'
DEMO_PATH = ROOT / 'phenotype' / 'demographics' / 'demographics.tsv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

TARGET_SEQ_LEN = 1024
SELECTED_TASKS = [
    'prolonged-vowel', 'glides-high-to-low', 'glides-low-to-high',
    'diadochokinesis-pataka', 'rainbow-passage', 'picture-description',
    'story-recall', 'maximum-phonation-time-1',
]

# ── 1. Model definition (must match training) ──
class ASTClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_base=False, attn_impl='eager'):
        super().__init__()
        if pretrained:
            self.ast = ASTModel.from_pretrained(
                'MIT/ast-finetuned-audioset-10-10-0.4593',
                attn_implementation=attn_impl
            )
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
        outputs = self.ast(input_values=x, output_attentions=True)
        return self.classifier(outputs.pooler_output), outputs.attentions

def resize_spectrogram(spec_arr, target_mel=128, target_time=1024):
    mel_ratio = target_mel / spec_arr.shape[0]
    time_ratio = target_time / spec_arr.shape[1]
    return zoom(spec_arr, (mel_ratio, time_ratio), order=1).astype(np.float32)

def process_spectrogram(spec_raw, target_len=1024):
    spec_arr = np.stack(spec_raw).astype(np.float32)
    n_mels, time_len = spec_arr.shape
    if time_len < target_len:
        spec_arr = np.pad(spec_arr, ((0, 0), (0, target_len - time_len)), mode='reflect')
    elif time_len > target_len:
        start = (time_len - target_len) // 2
        spec_arr = spec_arr[:, start:start + target_len]
    return spec_arr

# ── 2. Load data (sample for efficiency) ──
print('Loading data...')
pf = pq.ParquetFile(SPEC)
parts = []
for i in range(pf.num_row_groups):
    parts.append(pf.read_row_group(i, columns=['participant_id','session_id','task_name','mel_spectrogram','n_frames']).to_pandas())
spec = pd.concat(parts, ignore_index=True)
spec['participant_id'] = spec['participant_id'].astype(str).str.zfill(6)

pd_df = pd.read_csv(PD_PHEN, sep='\t')
ctrl_df = pd.read_csv(CTRL_PHEN, sep='\t')
pd_ids = set(pd_df['participant_id'].astype(str).str.zfill(6))
ctrl_ids = set(ctrl_df['participant_id'].astype(str).str.zfill(6)) - (set(pd_df['participant_id'].astype(str).str.zfill(6)) & set(ctrl_df['participant_id'].astype(str).str.zfill(6)))

spec['label'] = np.nan
spec.loc[spec['participant_id'].isin(pd_ids), 'label'] = 1
spec.loc[spec['participant_id'].isin(ctrl_ids), 'label'] = 0
data = spec.dropna(subset=['label']).copy()
data['label'] = data['label'].astype(int)
data = data[(data['task_name'].isin(SELECTED_TASKS)) & (data['n_frames'] >= 100)]

# Merge demographics
demo = pd.read_csv(DEMO_PATH, sep='\t')
demo['participant_id'] = demo['participant_id'].astype(str).str.zfill(6)
demo['age'] = pd.to_numeric(demo['age'], errors='coerce')
data = data.merge(demo[['participant_id', 'age']].drop_duplicates('participant_id'), on='participant_id', how='left')

# Sample: 1 recording per participant for prolonged-vowel (most common task)
task_data = data[data['task_name'] == 'prolonged-vowel'].copy()
# Take one recording per participant
sampled = task_data.groupby('participant_id').first().reset_index()
print(f'Sampled recordings for attention: {len(sampled)} (PD: {(sampled["label"]==1).sum()}, Ctrl: {(sampled["label"]==0).sum()})')

# ── 3. Load fold-1 model and extract attention ──
print('Loading model...')
model = ASTClassifier(num_classes=2, pretrained=True, attn_impl='eager').to(device)
model.load_state_dict(torch.load(str(RESULTS_DIR / 'ast_pd_v3_fold1.pt'), map_location=device))
model.eval()

# Load fold-1 normalization stats
cv_data = np.load(str(RESULTS_DIR / 'ast_pd_v3_cv_results.npz'), allow_pickle=True)
fold_mean = float(cv_data['norm_means'][0])
fold_std = float(cv_data['norm_stds'][0])

print('Extracting attention maps...')
attention_maps_pd = []
attention_maps_ctrl = []
spectrograms_pd = []
spectrograms_ctrl = []
ages_pd = []
ages_ctrl = []

for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc='Attention'):
    raw = process_spectrogram(row['mel_spectrogram'], TARGET_SEQ_LEN)
    resized = resize_spectrogram(raw)
    normed = (resized - fold_mean) / (fold_std + 1e-8)

    x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attentions = model(x)

    # Average attention across heads from last layer
    # attentions[-1] shape: (1, num_heads, seq_len, seq_len)
    last_attn = attentions[-1].squeeze(0).mean(dim=0)  # (seq_len, seq_len)

    # CLS token attention to all patches (first row, skip CLS+distill tokens)
    # AST has CLS token at position 0, distillation token at position 1
    cls_attn = last_attn[0, 2:].cpu().numpy()  # attention from CLS to patches

    if row['label'] == 1:
        attention_maps_pd.append(cls_attn)
        spectrograms_pd.append(resized)
        ages_pd.append(row.get('age', np.nan))
    else:
        attention_maps_ctrl.append(cls_attn)
        spectrograms_ctrl.append(resized)
        ages_ctrl.append(row.get('age', np.nan))

print(f'PD attention maps: {len(attention_maps_pd)}')
print(f'Control attention maps: {len(attention_maps_ctrl)}')

# ── 4. Reshape attention to spectrogram grid ──
# AST patch size: 16x16, input: 128x1024 = 8x64 patches = 512 patches
# Actual seq_len may differ due to positional embedding size
n_patches = len(attention_maps_pd[0])
print(f'Number of patches per sample: {n_patches}')

# Determine grid dimensions
# AST for 128x1024 with patch_size=16: freq_patches = 128/16 = 8, time_patches = 1024/16 = 64
freq_patches = 8
time_patches = n_patches // freq_patches
if freq_patches * time_patches != n_patches:
    # Try to find best factorization
    for fp in [8, 7, 6, 10, 12]:
        if n_patches % fp == 0:
            freq_patches = fp
            time_patches = n_patches // fp
            break
    print(f'Adjusted grid: {freq_patches} x {time_patches} = {freq_patches * time_patches}')

def reshape_attention(attn_vec, freq_p, time_p):
    """Reshape flat attention vector to 2D grid."""
    if len(attn_vec) > freq_p * time_p:
        attn_vec = attn_vec[:freq_p * time_p]
    elif len(attn_vec) < freq_p * time_p:
        attn_vec = np.pad(attn_vec, (0, freq_p * time_p - len(attn_vec)))
    return attn_vec.reshape(freq_p, time_p)

# Average attention maps
avg_attn_pd = np.mean([reshape_attention(a, freq_patches, time_patches) for a in attention_maps_pd], axis=0)
avg_attn_ctrl = np.mean([reshape_attention(a, freq_patches, time_patches) for a in attention_maps_ctrl], axis=0)
diff_attn = avg_attn_pd - avg_attn_ctrl

# ── 5. Age-matched attention (60-80 only) ──
ages_pd_arr = np.array(ages_pd)
ages_ctrl_arr = np.array(ages_ctrl)

am_pd_mask = (ages_pd_arr >= 60) & (ages_pd_arr <= 80)
am_ctrl_mask = (ages_ctrl_arr >= 60) & (ages_ctrl_arr <= 80)

if am_pd_mask.sum() > 0 and am_ctrl_mask.sum() > 0:
    avg_attn_pd_am = np.mean([reshape_attention(attention_maps_pd[i], freq_patches, time_patches)
                               for i in range(len(attention_maps_pd)) if am_pd_mask[i]], axis=0)
    avg_attn_ctrl_am = np.mean([reshape_attention(attention_maps_ctrl[i], freq_patches, time_patches)
                                 for i in range(len(attention_maps_ctrl)) if am_ctrl_mask[i]], axis=0)
    diff_attn_am = avg_attn_pd_am - avg_attn_ctrl_am
    has_am = True
    print(f'Age-matched attention: PD={am_pd_mask.sum()}, Ctrl={am_ctrl_mask.sum()}')
else:
    has_am = False
    print('Not enough age-matched samples for comparison')

# ── 6. Visualization ──
print('Creating figures...')

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# Row 1: Full cohort
im0 = axes[0, 0].imshow(avg_attn_pd, aspect='auto', cmap='hot', interpolation='bilinear')
axes[0, 0].set_title(f'PD Attention (N={len(attention_maps_pd)})', fontsize=11)
axes[0, 0].set_ylabel('Frequency patches')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(avg_attn_ctrl, aspect='auto', cmap='hot', interpolation='bilinear')
axes[0, 1].set_title(f'Control Attention (N={len(attention_maps_ctrl)})', fontsize=11)
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

vmax = max(abs(diff_attn.min()), abs(diff_attn.max()))
im2 = axes[0, 2].imshow(diff_attn, aspect='auto', cmap='RdBu_r', interpolation='bilinear',
                          vmin=-vmax, vmax=vmax)
axes[0, 2].set_title('PD - Control (Full Cohort)', fontsize=11)
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

# Row 2: Age-matched
if has_am:
    im3 = axes[1, 0].imshow(avg_attn_pd_am, aspect='auto', cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title(f'PD Attention [60-80] (N={am_pd_mask.sum()})', fontsize=11)
    axes[1, 0].set_ylabel('Frequency patches')
    axes[1, 0].set_xlabel('Time patches')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(avg_attn_ctrl_am, aspect='auto', cmap='hot', interpolation='bilinear')
    axes[1, 1].set_title(f'Control Attention [60-80] (N={am_ctrl_mask.sum()})', fontsize=11)
    axes[1, 1].set_xlabel('Time patches')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    vmax_am = max(abs(diff_attn_am.min()), abs(diff_attn_am.max()))
    im5 = axes[1, 2].imshow(diff_attn_am, aspect='auto', cmap='RdBu_r', interpolation='bilinear',
                              vmin=-vmax_am, vmax=vmax_am)
    axes[1, 2].set_title('PD - Control [60-80, Age-Matched]', fontsize=11)
    axes[1, 2].set_xlabel('Time patches')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
else:
    for ax in axes[1, :]:
        ax.set_visible(False)

plt.suptitle('AST Attention Maps for PD Classification (Prolonged Vowel, Last Layer, CLS Token)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

for ext in ['pdf', 'png']:
    fig.savefig(str(FIGURES_DIR / f'attention_maps_pd.{ext}'), dpi=200, bbox_inches='tight')
print(f'Saved: {FIGURES_DIR}/attention_maps_pd.pdf/.png')

# ── 7. Save numerical results ──
results = {
    'experiment': 'attention_map_analysis',
    'task': 'prolonged-vowel',
    'model': 'ast_pd_v3_fold1',
    'n_pd': len(attention_maps_pd),
    'n_ctrl': len(attention_maps_ctrl),
    'n_patches': int(n_patches),
    'grid_shape': [freq_patches, time_patches],
    'avg_attn_pd_mean': float(avg_attn_pd.mean()),
    'avg_attn_ctrl_mean': float(avg_attn_ctrl.mean()),
    'diff_attn_max': float(diff_attn.max()),
    'diff_attn_min': float(diff_attn.min()),
}

if has_am:
    results['n_pd_agematched'] = int(am_pd_mask.sum())
    results['n_ctrl_agematched'] = int(am_ctrl_mask.sum())
    results['diff_attn_am_max'] = float(diff_attn_am.max())
    results['diff_attn_am_min'] = float(diff_attn_am.min())

with open(str(RESULTS_DIR / 'attention_map_analysis.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save raw attention arrays for further analysis
np.savez(str(RESULTS_DIR / 'attention_maps.npz'),
    avg_attn_pd=avg_attn_pd,
    avg_attn_ctrl=avg_attn_ctrl,
    diff_attn=diff_attn,
    avg_attn_pd_am=avg_attn_pd_am if has_am else np.array([]),
    avg_attn_ctrl_am=avg_attn_ctrl_am if has_am else np.array([]),
)

print(f'Saved: {RESULTS_DIR}/attention_map_analysis.json')
print(f'Saved: {RESULTS_DIR}/attention_maps.npz')
print('DONE.')
