"""
Enhanced Interpretability — Statistical testing, multi-layer analysis,
and frequency-band mapping for AST attention maps in PD classification.

Builds on v3_PD_Interpretability.py with:
1. All 12 transformer layers (not just last)
2. Permutation test for PD vs Control attention difference
3. Frequency-band analysis with approximate Hz mapping
4. Cohen's d effect sizes per grid position
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.stats import ttest_ind
from transformers import ASTModel, ASTConfig
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

# ── 2. Load data ──
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
ctrl_ids = set(ctrl_df['participant_id'].astype(str).str.zfill(6)) - (pd_ids & set(ctrl_df['participant_id'].astype(str).str.zfill(6)))

spec['label'] = np.nan
spec.loc[spec['participant_id'].isin(pd_ids), 'label'] = 1
spec.loc[spec['participant_id'].isin(ctrl_ids), 'label'] = 0
data = spec.dropna(subset=['label']).copy()
data['label'] = data['label'].astype(int)

SELECTED_TASKS = [
    'prolonged-vowel', 'glides-high-to-low', 'glides-low-to-high',
    'diadochokinesis-pataka', 'rainbow-passage', 'picture-description',
    'story-recall', 'maximum-phonation-time-1',
]
data = data[(data['task_name'].isin(SELECTED_TASKS)) & (data['n_frames'] >= 100)]

# Merge demographics
demo = pd.read_csv(DEMO_PATH, sep='\t')
demo['participant_id'] = demo['participant_id'].astype(str).str.zfill(6)
demo['age'] = pd.to_numeric(demo['age'], errors='coerce')
data = data.merge(demo[['participant_id', 'age']].drop_duplicates('participant_id'), on='participant_id', how='left')

# Sample: 1 recording per participant for prolonged-vowel
task_data = data[data['task_name'] == 'prolonged-vowel'].copy()
sampled = task_data.groupby('participant_id').first().reset_index()
print(f'Sampled: {len(sampled)} (PD: {(sampled["label"]==1).sum()}, Ctrl: {(sampled["label"]==0).sum()})')

# ── 3. Load model and extract ALL-LAYER attention ──
print('Loading model...')
model = ASTClassifier(num_classes=2, pretrained=True, attn_impl='eager').to(device)
model.load_state_dict(torch.load(str(RESULTS_DIR / 'ast_pd_v3_fold1.pt'), map_location=device))
model.eval()

cv_data = np.load(str(RESULTS_DIR / 'ast_pd_v3_cv_results.npz'), allow_pickle=True)
fold_mean = float(cv_data['norm_means'][0])
fold_std = float(cv_data['norm_stds'][0])

N_LAYERS = 12
print(f'Extracting attention from all {N_LAYERS} layers...')

# Store per-sample attention for all layers: [layer][sample] = cls_attn_vector
all_layer_attn_pd = [[] for _ in range(N_LAYERS)]
all_layer_attn_ctrl = [[] for _ in range(N_LAYERS)]
ages_pd = []
ages_ctrl = []

for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc='Attention'):
    raw = process_spectrogram(row['mel_spectrogram'], TARGET_SEQ_LEN)
    resized = resize_spectrogram(raw)
    normed = (resized - fold_mean) / (fold_std + 1e-8)

    x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attentions = model(x)

    for layer_idx in range(N_LAYERS):
        # Average across heads, CLS token attention to patches (skip CLS+distill)
        layer_attn = attentions[layer_idx].squeeze(0).mean(dim=0)  # (seq_len, seq_len)
        cls_attn = layer_attn[0, 2:].cpu().numpy()  # CLS -> patches

        if row['label'] == 1:
            all_layer_attn_pd[layer_idx].append(cls_attn)
        else:
            all_layer_attn_ctrl[layer_idx].append(cls_attn)

    if row['label'] == 1:
        ages_pd.append(row.get('age', np.nan))
    else:
        ages_ctrl.append(row.get('age', np.nan))

n_pd = len(all_layer_attn_pd[0])
n_ctrl = len(all_layer_attn_ctrl[0])
n_patches = len(all_layer_attn_pd[0][0])
print(f'PD: {n_pd}, Control: {n_ctrl}, Patches: {n_patches}')

# ── 4. Grid shape ──
freq_patches = 6
time_patches = n_patches // freq_patches
assert freq_patches * time_patches == n_patches, f'Grid mismatch: {freq_patches}*{time_patches} != {n_patches}'
print(f'Grid: {freq_patches} x {time_patches}')

def reshape_attention(attn_vec, fp, tp):
    if len(attn_vec) > fp * tp:
        attn_vec = attn_vec[:fp * tp]
    elif len(attn_vec) < fp * tp:
        attn_vec = np.pad(attn_vec, (0, fp * tp - len(attn_vec)))
    return attn_vec.reshape(fp, tp)

# ── 5. Multi-layer analysis ──
print('Computing per-layer statistics...')

layer_divergence = []  # Mean absolute difference PD-Ctrl per layer
layer_pd_means = []
layer_ctrl_means = []

for layer_idx in range(N_LAYERS):
    pd_maps = np.array([reshape_attention(a, freq_patches, time_patches) for a in all_layer_attn_pd[layer_idx]])
    ctrl_maps = np.array([reshape_attention(a, freq_patches, time_patches) for a in all_layer_attn_ctrl[layer_idx]])

    avg_pd = pd_maps.mean(axis=0)
    avg_ctrl = ctrl_maps.mean(axis=0)
    diff = avg_pd - avg_ctrl

    layer_pd_means.append(avg_pd)
    layer_ctrl_means.append(avg_ctrl)
    layer_divergence.append(np.abs(diff).mean())

    print(f'  Layer {layer_idx+1:2d}: PD mean={avg_pd.mean():.6f}, Ctrl mean={avg_ctrl.mean():.6f}, |diff|={np.abs(diff).mean():.6f}')

# ── 6. Statistical testing on last layer ──
print('\nStatistical testing (last layer)...')
last_layer_idx = N_LAYERS - 1
pd_maps_last = np.array([reshape_attention(a, freq_patches, time_patches) for a in all_layer_attn_pd[last_layer_idx]])
ctrl_maps_last = np.array([reshape_attention(a, freq_patches, time_patches) for a in all_layer_attn_ctrl[last_layer_idx]])

# Pixel-wise t-test
t_stats = np.zeros((freq_patches, time_patches))
p_values = np.zeros((freq_patches, time_patches))
cohens_d = np.zeros((freq_patches, time_patches))

for i in range(freq_patches):
    for j in range(time_patches):
        pd_vals = pd_maps_last[:, i, j]
        ctrl_vals = ctrl_maps_last[:, i, j]
        t, p = ttest_ind(pd_vals, ctrl_vals, equal_var=False)
        t_stats[i, j] = t
        p_values[i, j] = p
        # Cohen's d
        pooled_std = np.sqrt((pd_vals.std()**2 + ctrl_vals.std()**2) / 2)
        if pooled_std > 0:
            cohens_d[i, j] = (pd_vals.mean() - ctrl_vals.mean()) / pooled_std

# Multiple testing correction (Bonferroni)
n_tests = freq_patches * time_patches
sig_bonferroni = p_values < (0.05 / n_tests)
# Also FDR (Benjamini-Hochberg)
flat_p = p_values.flatten()
sorted_idx = np.argsort(flat_p)
fdr_threshold = np.zeros_like(flat_p, dtype=bool)
m = len(flat_p)
for rank, idx in enumerate(sorted_idx, 1):
    if flat_p[idx] <= 0.05 * rank / m:
        fdr_threshold[idx] = True
    else:
        break
sig_fdr = fdr_threshold.reshape(freq_patches, time_patches)

print(f'Significant positions (Bonferroni p<0.05/{n_tests}): {sig_bonferroni.sum()}/{n_tests}')
print(f'Significant positions (FDR q<0.05): {sig_fdr.sum()}/{n_tests}')
print(f'Cohen\'s d range: [{cohens_d.min():.3f}, {cohens_d.max():.3f}]')
print(f'Mean |Cohen\'s d|: {np.abs(cohens_d).mean():.3f}')

# ── 7. Permutation test (global) ──
print('\nPermutation test (N=1000)...')
all_maps = np.concatenate([pd_maps_last, ctrl_maps_last], axis=0)
true_diff = np.abs(pd_maps_last.mean(axis=0) - ctrl_maps_last.mean(axis=0)).mean()

n_perm = 1000
perm_diffs = np.zeros(n_perm)
rng = np.random.RandomState(42)
n_total = len(all_maps)

for p in range(n_perm):
    perm_idx = rng.permutation(n_total)
    perm_pd = all_maps[perm_idx[:n_pd]]
    perm_ctrl = all_maps[perm_idx[n_pd:]]
    perm_diffs[p] = np.abs(perm_pd.mean(axis=0) - perm_ctrl.mean(axis=0)).mean()

perm_p = (perm_diffs >= true_diff).sum() / n_perm
print(f'True mean |diff|: {true_diff:.6f}')
print(f'Permutation p-value: {perm_p:.4f}')
print(f'Permutation null mean: {perm_diffs.mean():.6f} (SD {perm_diffs.std():.6f})')

# ── 8. Frequency-band analysis ──
# AST patch_size=16, frequency_stride=10, input freq=128 mel bins
# Patch centers: 8, 18, 28, 38, ...; with stride=10, 6 patches cover 0-128 mel bins
# Approximate mapping: mel bin ranges per patch row
# Row 0: mel bins 0-15 (lowest), Row 5: mel bins ~50-65+ (highest)
# For 128 mel bins spanning ~0-8000 Hz (typical mel scale):
freq_labels = ['0-21', '10-31', '20-41', '30-51', '40-61', '50-71']
freq_hz_approx = ['~0-600', '~200-1200', '~500-2000', '~1000-3500', '~2000-5500', '~3500-8000+']

print('\nFrequency-band attention (last layer):')
diff_last = pd_maps_last.mean(axis=0) - ctrl_maps_last.mean(axis=0)

freq_band_stats = []
for i in range(freq_patches):
    pd_band = pd_maps_last[:, i, :].mean(axis=1)  # mean across time per sample
    ctrl_band = ctrl_maps_last[:, i, :].mean(axis=1)
    t, p = ttest_ind(pd_band, ctrl_band, equal_var=False)
    d = (pd_band.mean() - ctrl_band.mean()) / np.sqrt((pd_band.std()**2 + ctrl_band.std()**2) / 2) if (pd_band.std() + ctrl_band.std()) > 0 else 0

    freq_band_stats.append({
        'band': i,
        'mel_bins': freq_labels[i],
        'hz_approx': freq_hz_approx[i],
        'pd_mean': float(pd_band.mean()),
        'ctrl_mean': float(ctrl_band.mean()),
        'diff': float(pd_band.mean() - ctrl_band.mean()),
        't_stat': float(t),
        'p_value': float(p),
        'cohens_d': float(d),
    })
    print(f'  Band {i} (mel {freq_labels[i]}, {freq_hz_approx[i]}): PD={pd_band.mean():.6f}, Ctrl={ctrl_band.mean():.6f}, d={d:.3f}, p={p:.4f}')

# ── 9. Age-matched analysis ──
ages_pd_arr = np.array(ages_pd)
ages_ctrl_arr = np.array(ages_ctrl)
am_pd_mask = (ages_pd_arr >= 60) & (ages_pd_arr <= 80)
am_ctrl_mask = (ages_ctrl_arr >= 60) & (ages_ctrl_arr <= 80)

if am_pd_mask.sum() > 0 and am_ctrl_mask.sum() > 0:
    print(f'\nAge-matched [60-80]: PD={am_pd_mask.sum()}, Ctrl={am_ctrl_mask.sum()}')

    am_pd_maps = pd_maps_last[am_pd_mask]
    am_ctrl_maps = ctrl_maps_last[am_ctrl_mask]

    # Permutation test on age-matched
    am_all = np.concatenate([am_pd_maps, am_ctrl_maps], axis=0)
    am_true_diff = np.abs(am_pd_maps.mean(axis=0) - am_ctrl_maps.mean(axis=0)).mean()
    am_n_pd = am_pd_mask.sum()

    am_perm_diffs = np.zeros(n_perm)
    for p in range(n_perm):
        perm_idx = rng.permutation(len(am_all))
        perm_pd = am_all[perm_idx[:am_n_pd]]
        perm_ctrl = am_all[perm_idx[am_n_pd:]]
        am_perm_diffs[p] = np.abs(perm_pd.mean(axis=0) - perm_ctrl.mean(axis=0)).mean()

    am_perm_p = (am_perm_diffs >= am_true_diff).sum() / n_perm
    print(f'Age-matched permutation p-value: {am_perm_p:.4f}')

    # Cohen's d for age-matched
    am_cohens_d = np.zeros((freq_patches, time_patches))
    for i in range(freq_patches):
        for j in range(time_patches):
            pd_v = am_pd_maps[:, i, j]
            ctrl_v = am_ctrl_maps[:, i, j]
            ps = np.sqrt((pd_v.std()**2 + ctrl_v.std()**2) / 2)
            am_cohens_d[i, j] = (pd_v.mean() - ctrl_v.mean()) / ps if ps > 0 else 0

    print(f'Age-matched Cohen\'s d range: [{am_cohens_d.min():.3f}, {am_cohens_d.max():.3f}]')
    print(f'Age-matched mean |d|: {np.abs(am_cohens_d).mean():.3f}')

# ── 10. Generate enhanced figure ──
print('\nCreating enhanced figure...')

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Row 1: Attention maps (same as original but with significance overlay)
avg_pd = pd_maps_last.mean(axis=0)
avg_ctrl = ctrl_maps_last.mean(axis=0)
diff_map = avg_pd - avg_ctrl

ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(avg_pd, aspect='auto', cmap='hot', interpolation='bilinear')
ax1.set_title(f'PD Attention (N={n_pd})', fontsize=11)
ax1.set_ylabel('Frequency patch')
plt.colorbar(im1, ax=ax1, fraction=0.046)

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(avg_ctrl, aspect='auto', cmap='hot', interpolation='bilinear')
ax2.set_title(f'Control Attention (N={n_ctrl})', fontsize=11)
plt.colorbar(im2, ax=ax2, fraction=0.046)

vmax = max(abs(diff_map.min()), abs(diff_map.max()))
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(diff_map, aspect='auto', cmap='RdBu_r', interpolation='bilinear',
                  vmin=-vmax, vmax=vmax)
ax3.set_title('PD - Control Difference', fontsize=11)
plt.colorbar(im3, ax=ax3, fraction=0.046)

# Row 2: Cohen's d map + layer divergence + frequency band
ax4 = fig.add_subplot(gs[1, 0])
dmax = max(abs(cohens_d.min()), abs(cohens_d.max()))
im4 = ax4.imshow(cohens_d, aspect='auto', cmap='RdBu_r', interpolation='bilinear',
                  vmin=-dmax, vmax=dmax)
ax4.set_title("Cohen's d (Full Cohort)", fontsize=11)
ax4.set_ylabel('Frequency patch')
plt.colorbar(im4, ax=ax4, fraction=0.046, label="Cohen's d")

ax5 = fig.add_subplot(gs[1, 1])
layers = np.arange(1, N_LAYERS + 1)
ax5.bar(layers, layer_divergence, color='steelblue', edgecolor='navy', alpha=0.8)
ax5.set_xlabel('Transformer Layer')
ax5.set_ylabel('Mean |PD - Control| Attention')
ax5.set_title('Layer-wise Attention Divergence', fontsize=11)
ax5.set_xticks(layers)
# Mark the layer with max divergence
max_layer = np.argmax(layer_divergence) + 1
ax5.bar(max_layer, layer_divergence[max_layer-1], color='firebrick', edgecolor='darkred', alpha=0.9)

ax6 = fig.add_subplot(gs[1, 2])
band_diffs = [s['diff'] for s in freq_band_stats]
band_ps = [s['p_value'] for s in freq_band_stats]
colors = ['firebrick' if p < 0.05 else 'steelblue' for p in band_ps]
bars = ax6.barh(range(freq_patches), band_diffs, color=colors, edgecolor='black', alpha=0.8)
ax6.set_yticks(range(freq_patches))
ax6.set_yticklabels([f'{freq_hz_approx[i]}' for i in range(freq_patches)], fontsize=9)
ax6.set_xlabel('Mean Attention Diff (PD - Control)')
ax6.set_title('Frequency-Band Analysis', fontsize=11)
ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
# Add significance markers
for i, p in enumerate(band_ps):
    if p < 0.001:
        ax6.text(band_diffs[i], i, ' ***', va='center', fontsize=10, fontweight='bold')
    elif p < 0.01:
        ax6.text(band_diffs[i], i, ' **', va='center', fontsize=10, fontweight='bold')
    elif p < 0.05:
        ax6.text(band_diffs[i], i, ' *', va='center', fontsize=10, fontweight='bold')

# Row 3: Age-matched Cohen's d + age-matched difference + permutation distribution
if am_pd_mask.sum() > 0 and am_ctrl_mask.sum() > 0:
    am_diff = am_pd_maps.mean(axis=0) - am_ctrl_maps.mean(axis=0)

    ax7 = fig.add_subplot(gs[2, 0])
    am_dmax = max(abs(am_cohens_d.min()), abs(am_cohens_d.max()))
    im7 = ax7.imshow(am_cohens_d, aspect='auto', cmap='RdBu_r', interpolation='bilinear',
                      vmin=-am_dmax, vmax=am_dmax)
    ax7.set_title(f"Cohen's d [60-80] (N={am_pd_mask.sum()}+{am_ctrl_mask.sum()})", fontsize=11)
    ax7.set_ylabel('Frequency patch')
    ax7.set_xlabel('Time patch')
    plt.colorbar(im7, ax=ax7, fraction=0.046, label="Cohen's d")

    ax8 = fig.add_subplot(gs[2, 1])
    am_vmax = max(abs(am_diff.min()), abs(am_diff.max()))
    im8 = ax8.imshow(am_diff, aspect='auto', cmap='RdBu_r', interpolation='bilinear',
                      vmin=-am_vmax, vmax=am_vmax)
    ax8.set_title(f'PD - Control [60-80, Age-Matched]', fontsize=11)
    ax8.set_xlabel('Time patch')
    plt.colorbar(im8, ax=ax8, fraction=0.046)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.hist(perm_diffs, bins=40, color='lightgray', edgecolor='gray', alpha=0.8, label='Null (full)')
    ax9.axvline(true_diff, color='firebrick', linewidth=2, linestyle='-', label=f'Observed (p={perm_p:.3f})')
    ax9.hist(am_perm_diffs, bins=40, color='lightblue', edgecolor='steelblue', alpha=0.5, label='Null (age-matched)')
    ax9.axvline(am_true_diff, color='navy', linewidth=2, linestyle='--', label=f'Observed AM (p={am_perm_p:.3f})')
    ax9.set_xlabel('Mean |PD - Control| Attention')
    ax9.set_ylabel('Count')
    ax9.set_title('Permutation Test (N=1000)', fontsize=11)
    ax9.legend(fontsize=8)

plt.suptitle('Enhanced AST Attention Analysis for PD Classification\n(Prolonged Vowel, CLS Token, Fold 1)',
             fontsize=14, fontweight='bold')

for ext in ['pdf', 'png']:
    fig.savefig(str(FIGURES_DIR / f'attention_maps_enhanced.{ext}'), dpi=200, bbox_inches='tight')
print(f'Saved: {FIGURES_DIR}/attention_maps_enhanced.pdf/.png')

# ── 11. Save results ──
results = {
    'experiment': 'enhanced_attention_analysis',
    'task': 'prolonged-vowel',
    'model': 'ast_pd_v3_fold1',
    'n_pd': n_pd,
    'n_ctrl': n_ctrl,
    'n_patches': int(n_patches),
    'grid_shape': [freq_patches, time_patches],
    'n_layers': N_LAYERS,
    'layer_divergence': [float(d) for d in layer_divergence],
    'max_divergence_layer': int(np.argmax(layer_divergence) + 1),
    'last_layer_stats': {
        'cohens_d_range': [float(cohens_d.min()), float(cohens_d.max())],
        'mean_abs_cohens_d': float(np.abs(cohens_d).mean()),
        'n_sig_bonferroni': int(sig_bonferroni.sum()),
        'n_sig_fdr': int(sig_fdr.sum()),
        'n_tests': int(n_tests),
        'permutation_p': float(perm_p),
        'permutation_true_diff': float(true_diff),
        'permutation_null_mean': float(perm_diffs.mean()),
    },
    'frequency_band_stats': freq_band_stats,
}

if am_pd_mask.sum() > 0 and am_ctrl_mask.sum() > 0:
    results['age_matched'] = {
        'n_pd': int(am_pd_mask.sum()),
        'n_ctrl': int(am_ctrl_mask.sum()),
        'permutation_p': float(am_perm_p),
        'permutation_true_diff': float(am_true_diff),
        'cohens_d_range': [float(am_cohens_d.min()), float(am_cohens_d.max())],
        'mean_abs_cohens_d': float(np.abs(am_cohens_d).mean()),
    }

with open(str(RESULTS_DIR / 'attention_enhanced_analysis.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save arrays
np.savez(str(RESULTS_DIR / 'attention_enhanced.npz'),
    layer_divergence=np.array(layer_divergence),
    cohens_d=cohens_d,
    p_values=p_values,
    t_stats=t_stats,
    am_cohens_d=am_cohens_d if am_pd_mask.sum() > 0 else np.array([]),
    perm_diffs=perm_diffs,
    am_perm_diffs=am_perm_diffs if am_pd_mask.sum() > 0 else np.array([]),
)

print(f'Saved: {RESULTS_DIR}/attention_enhanced_analysis.json')
print(f'Saved: {RESULTS_DIR}/attention_enhanced.npz')
print('DONE.')
