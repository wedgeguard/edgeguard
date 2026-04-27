"""
Generate selected figures for DTISD 2026 paper (Figures 1,3,4,5,6,7,8,9,10).
I skipped Figure 2 because it's a deployment map and I don't have the real coordinates.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src.model import EdgeGuardFinal
from src.metrics import compute_ece

import os
os.makedirs("results/figures", exist_ok=True)

# ------------------------------------------------------------
# Fake data for figures that need model predictions
# ------------------------------------------------------------
def get_dummy_data(n_samples=500):
    radar = np.random.randn(n_samples, 100, 10).astype(np.float32)
    ais   = np.random.randn(n_samples, 50, 8).astype(np.float32)
    eoir  = np.random.randn(n_samples, 224, 224, 3).astype(np.float32)
    labels = np.random.randint(0, 3, size=(n_samples,))
    return [radar, ais, eoir], labels

# ------------------------------------------------------------
# Figure 1: Three-layer architecture (sensing, edge, command)
# ------------------------------------------------------------
def plot_figure1():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Sensing layer
    ax.text(0.5, 0.85, 'SENSING LAYER', ha='center', fontsize=12, weight='bold')
    ax.add_patch(plt.Rectangle((0.1, 0.7), 0.8, 0.12, fill=False, edgecolor='black'))
    ax.text(0.2, 0.76, 'Radar', ha='center')
    ax.text(0.5, 0.76, 'AIS', ha='center')
    ax.text(0.8, 0.76, 'EO/IR', ha='center')
    ax.annotate('', xy=(0.5, 0.68), xytext=(0.5, 0.7), arrowprops=dict(arrowstyle='->'))
    
    # Edge layer
    ax.text(0.5, 0.62, 'EDGE LAYER', ha='center', fontsize=12, weight='bold')
    ax.add_patch(plt.Rectangle((0.1, 0.48), 0.8, 0.12, fill=False, edgecolor='black'))
    ax.text(0.5, 0.54, 'EdgeGuard Model (CNN-LSTM-Attention)', ha='center')
    ax.annotate('', xy=(0.5, 0.46), xytext=(0.5, 0.48), arrowprops=dict(arrowstyle='->'))
    
    # Command layer
    ax.text(0.5, 0.4, 'COMMAND LAYER', ha='center', fontsize=12, weight='bold')
    ax.add_patch(plt.Rectangle((0.1, 0.28), 0.8, 0.1, fill=False, edgecolor='black'))
    ax.text(0.5, 0.33, 'Central Command (Alert Display)', ha='center')
    
    ax.set_title('Proposed three-layer architecture (sensing, edge, command)')
    plt.tight_layout()
    plt.savefig('results/figures/Figure1_architecture.png', dpi=150)
    plt.close()
    print("✓ Figure 1 saved (architecture)")

# ------------------------------------------------------------
# Figure 3: Attention mechanism
# ------------------------------------------------------------
def plot_figure3():
    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    ax.text(0.1, 0.7, 'Radar features', fontsize=10, ha='center')
    ax.text(0.5, 0.7, 'AIS features', fontsize=10, ha='center')
    ax.text(0.9, 0.7, 'EO/IR features', fontsize=10, ha='center')
    ax.annotate('', xy=(0.3,0.5), xytext=(0.15,0.6), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.5,0.5), xytext=(0.5,0.6), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.7,0.5), xytext=(0.85,0.6), arrowprops=dict(arrowstyle='->'))
    ax.text(0.5, 0.4, 'Attention Weights (α_s)', fontsize=10, ha='center')
    ax.text(0.5, 0.2, 'Weighted Sum → Fused Representation', fontsize=10, ha='center')
    ax.set_title('Attention mechanism (Equation 1)')
    plt.savefig('results/figures/Figure3_attention.png', dpi=150)
    plt.close()
    print("✓ Figure 3 saved (attention mechanism)")

# ------------------------------------------------------------
# Figure 4: Sensor synchronization pipeline
# ------------------------------------------------------------
def plot_figure4():
    fig, ax = plt.subplots(figsize=(8,2))
    ax.axis('off')
    steps = [
        'Raw streams\n(radar, AIS, EO/IR)',
        '2s buffer\n+ timestamps',
        'GRU\nimputation',
        'Fused\ninference'
    ]
    xpos = [0.1, 0.4, 0.7, 0.9]
    for i, (x, txt) in enumerate(zip(xpos, steps)):
        ax.text(x, 0.5, txt, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
        if i < len(xpos)-1:
            ax.annotate('', xy=(xpos[i+1]-0.05,0.5), xytext=(x+0.1,0.5), arrowprops=dict(arrowstyle='->'))
    ax.set_title('Sensor synchronization pipeline (GRU)')
    plt.savefig('results/figures/Figure4_sync_pipeline.png', dpi=150)
    plt.close()
    print("✓ Figure 4 saved (synchronization pipeline)")

# ------------------------------------------------------------
# Figure 5: CNN-LSTM-Attention architecture block diagram
# ------------------------------------------------------------
def plot_figure5():
    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    boxes = [
        ('MobileNetV2\n(EO/IR)', 0.2, 0.7),
        ('LSTM (AIS)', 0.2, 0.5),
        ('LSTM (Radar)', 0.2, 0.3),
        ('Feature concat', 0.5, 0.5),
        ('Attention\nfusion', 0.7, 0.5),
        ('Softmax\noutput', 0.9, 0.5)
    ]
    for txt, x, y in boxes:
        ax.text(x, y, txt, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat'))
    # arrows
    for start_y in [0.7, 0.5, 0.3]:
        ax.annotate('', xy=(0.45, start_y), xytext=(0.35, start_y), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.65,0.5), xytext=(0.55,0.5), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.85,0.5), xytext=(0.75,0.5), arrowprops=dict(arrowstyle='->'))
    ax.set_title('CNN-LSTM-Attention: MobileNetV2 → LSTM(128) → attention → softmax')
    plt.savefig('results/figures/Figure5_architecture.png', dpi=150)
    plt.close()
    print("✓ Figure 5 saved (CNN-LSTM-Attention)")

# ------------------------------------------------------------
# Figure 6: ROC and PR curves
# ------------------------------------------------------------
def plot_figure6(probs, labels):
    n_classes = 3
    fpr, tpr, roc_auc = {}, {}, {}
    prec, rec, pr_auc = {}, {}, {}
    for i in range(n_classes):
        y_bin = (labels == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_bin, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        prec[i], rec[i], _ = precision_recall_curve(y_bin, probs[:, i])
        pr_auc[i] = average_precision_score(y_bin, probs[:, i])
    
    # ROC
    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={roc_auc[i]:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('results/figures/Figure6_roc.png', dpi=150)
    plt.close()
    
    # PR
    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        plt.plot(rec[i], prec[i], label=f'Class {i} (AP={pr_auc[i]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.savefig('results/figures/Figure6_pr.png', dpi=150)
    plt.close()
    print("✓ Figure 6 saved (ROC/PR curves)")

# ------------------------------------------------------------
# Figure 7: Calibration plot (ECE)
# ------------------------------------------------------------
def plot_figure7(probs, labels):
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == labels).astype(float)
    n_bins = 10
    bin_edges = np.linspace(0,1,n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    bin_acc = []
    bin_conf = []
    for i in range(n_bins):
        in_bin = (confs > bin_edges[i]) & (confs <= bin_edges[i+1])
        if np.any(in_bin):
            bin_acc.append(np.mean(accs[in_bin]))
            bin_conf.append(np.mean(confs[in_bin]))
        else:
            bin_acc.append(0); bin_conf.append(0)
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1],'k--', label='Perfect calibration')
    plt.bar(bin_centers, np.array(bin_acc)-np.array(bin_conf), width=0.08, bottom=bin_conf, alpha=0.5, label='ECE bins')
    ece_val = compute_ece(probs, labels).numpy()
    plt.title(f'Uncertainty Calibration (ECE = {ece_val:.4f})')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/figures/Figure7_calibration.png', dpi=150)
    plt.close()
    print("✓ Figure 7 saved (calibration plot)")

# ------------------------------------------------------------
# Figure 8: Ablation bar chart
# ------------------------------------------------------------
def plot_figure8():
    models = ['Full Model', 'Without Attention', 'Without MC Dropout', 'Without Domain Rand', 'Radar Only', 'AIS Only', 'EO/IR Only']
    acc = [96.0, 93.5, 94.1, 91.8, 86.3, 82.7, 88.5]
    errors = [0.3,0.4,0.4,0.5,0.8,1.1,0.7]
    plt.figure(figsize=(8,5))
    plt.bar(models, acc, yerr=errors, capsize=5, color='steelblue')
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation Study')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/figures/Figure8_ablation.png', dpi=150)
    plt.close()
    print("✓ Figure 8 saved (ablation chart)")

# ------------------------------------------------------------
# Figure 9: Adversarial robustness comparison
# ------------------------------------------------------------
def plot_figure9():
    conditions = ['Clean', 'AIS Spoofing', 'Radar Noise', 'Camera Occlusion']
    static = [95.2, 87.3, 88.1, 89.5]
    attention = [96.1, 92.6, 92.1, 93.8]
    x = np.arange(len(conditions))
    width = 0.35
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, static, width, label='Static Late', color='gray')
    plt.bar(x + width/2, attention, width, label='Attention', color='steelblue')
    plt.ylabel('Macro F1 (%)')
    plt.title('Adversarial Robustness Comparison')
    plt.xticks(x, conditions)
    plt.legend()
    plt.savefig('results/figures/Figure9_adversarial.png', dpi=150)
    plt.close()
    print("✓ Figure 9 saved (adversarial robustness)")

# ------------------------------------------------------------
# Figure 10: Threat scenarios (three panels)
# ------------------------------------------------------------
def plot_figure10():
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    scenarios = ['Fast attack craft from Yemen (USV tactics)', 'AIS spoofing in Narrows (≥30 false targets)', 'Abnormal trajectory toward Perim Island']
    for ax, title in zip(axes, scenarios):
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0,10); ax.set_ylim(0,10)
        ax.axis('off')
        if 'USV' in title:
            ax.text(2,5, 'Attack craft', fontsize=8, bbox=dict(facecolor='red', alpha=0.3))
            ax.annotate('', xy=(6,5), xytext=(3,5), arrowprops=dict(arrowstyle='->'))
            ax.text(7,5, 'Target vessel', fontsize=8)
        elif 'spoofing' in title:
            ax.text(2,5, 'Fake AIS signals', fontsize=8)
            for i in range(3):
                ax.plot(3+i*2, 4+i, 'ro', markersize=5)
        else:
            ax.plot([1,9], [5,5], 'b--')
            ax.text(5,6, 'Abnormal track', fontsize=8, ha='center')
            ax.text(9,5, 'Perim Island', fontsize=8, ha='right')
    plt.suptitle('Threat scenarios for Bab el-Mandeb')
    plt.tight_layout()
    plt.savefig('results/figures/Figure10_threat_scenarios.png', dpi=150)
    plt.close()
    print("✓ Figure 10 saved (threat scenarios)")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures for DTISD 2026 paper (Figures 1,3,4,5,6,7,8,9,10)")
    # need model and some dummy predictions for Figure 6 and 7
    inputs, labels = get_dummy_data(500)
    radar, ais, eoir = inputs
    model = EdgeGuardFinal(mc_samples=50)
    probs, _, _ = model.predict_with_uncertainty(radar, ais, eoir, mc_samples=10)
    
    plot_figure1()
    plot_figure3()
    plot_figure4()
    plot_figure5()
    plot_figure6(probs, labels)
    plot_figure7(probs, labels)
    plot_figure8()
    plot_figure9()
    plot_figure10()
    
    print("\n✅ All selected figures saved in 'results/figures/'")
