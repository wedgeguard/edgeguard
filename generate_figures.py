"""
Generate plots for the paper (Figure 7, 9, 11, etc.)
I know it's a bit messy, but it works.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src.model import EdgeGuardFinal
from src.metrics import compute_ece

# make sure results folder exists
import os
os.makedirs("results", exist_ok=True)

# -------------------------------
# Helper: calibration plot (Figure 7)
# -------------------------------
def plot_calibration(probs, labels, n_bins=10):
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == labels).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_acc = []
    bin_conf = []
    for i in range(n_bins):
        in_bin = (confs > bin_edges[i]) & (confs <= bin_edges[i+1])
        if np.any(in_bin):
            bin_acc.append(np.mean(accs[in_bin]))
            bin_conf.append(np.mean(confs[in_bin]))
        else:
            bin_acc.append(0)
            bin_conf.append(0)
    
    plt.figure(figsize=(6,6))
    plt.plot([0,1], [0,1], 'k--', label='perfect calibration')
    plt.bar(bin_centers, np.array(bin_acc) - np.array(bin_conf),
            width=0.08, bottom=bin_conf, alpha=0.5, label='ECE bins')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    ece_val = compute_ece(probs, labels).numpy()
    plt.title(f'Uncertainty Calibration (ECE = {ece_val:.4f})')
    plt.legend()
    plt.savefig('results/calibration_plot.png', dpi=150)
    plt.close()
    print("Saved calibration plot to results/calibration_plot.png")

# -------------------------------
# Helper: attention weights (Figure 9)
# -------------------------------
def plot_attention_weights(model, sample_input):
    radar, ais, eoir = sample_input
    # try to get the attention layer (its name is 'modality_attention')
    att_model = tf.keras.Model(
        inputs=model.model.input,
        outputs=model.model.get_layer('modality_attention').output
    )
    weights = att_model.predict([radar, ais, eoir])
    w_mean = np.mean(weights, axis=0)
    sensors = ['Radar', 'AIS', 'EO/IR']
    
    plt.figure(figsize=(6,4))
    plt.bar(sensors, w_mean, color=['navy', 'darkorange', 'green'])
    plt.ylabel('Attention weight')
    plt.title('Modality attention (normal operation)')
    for i, v in enumerate(w_mean):
        plt.text(i, v+0.02, f'{v:.2f}', ha='center')
    plt.savefig('results/attention_shift.png', dpi=150)
    plt.close()
    print("Saved attention plot to results/attention_shift.png")

# -------------------------------
# Helper: ROC and PR curves (Figure 11)
# -------------------------------
def plot_roc_pr(probs, labels):
    n_classes = 3
    fpr = {}
    tpr = {}
    roc_auc = {}
    prec = {}
    rec = {}
    pr_auc = {}
    
    for i in range(n_classes):
        y_bin = (labels == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_bin, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        prec[i], rec[i], _ = precision_recall_curve(y_bin, probs[:, i])
        pr_auc[i] = average_precision_score(y_bin, probs[:, i])
    
    # ROC plot
    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={roc_auc[i]:.3f})')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('results/roc_curves.png', dpi=150)
    plt.close()
    
    # PR plot
    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        plt.plot(rec[i], prec[i], label=f'Class {i} (AP={pr_auc[i]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.savefig('results/pr_curves.png', dpi=150)
    plt.close()
    print("Saved ROC and PR curves to results/")

# -------------------------------
# Helper: confusion matrix heatmap (optional)
# -------------------------------
def plot_confusion_matrix(cm, classes=['Normal','Suspicious','Hostile']):
    import seaborn as sns
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved confusion matrix to results/confusion_matrix.png")

# -------------------------------
# Main: generate all figures using dummy data (just to test)
# TODO: replace dummy data with real test set later
# -------------------------------
if __name__ == "__main__":
    print("Generating figures using dummy data (real test set not yet loaded)")
    model = EdgeGuardFinal(mc_samples=50)
    N = 500
    radar_dummy = np.random.randn(N, 100, 10).astype(np.float32)
    ais_dummy   = np.random.randn(N, 50, 8).astype(np.float32)
    eoir_dummy  = np.random.randn(N, 224, 224, 3).astype(np.float32)
    labels_dummy = np.random.randint(0, 3, size=(N,))
    
    # get predictions
    probs, _, _ = model.predict_with_uncertainty(radar_dummy, ais_dummy, eoir_dummy, mc_samples=10)
    
    plot_calibration(probs, labels_dummy)
    plot_attention_weights(model, [radar_dummy[:1], ais_dummy[:1], eoir_dummy[:1]])
    plot_roc_pr(probs, labels_dummy)
    preds = np.argmax(probs, axis=1)
    cm = tf.math.confusion_matrix(labels_dummy, preds).numpy()
    plot_confusion_matrix(cm)
    print("\nAll figures saved. (they are based on random data, not actual paper results)")
