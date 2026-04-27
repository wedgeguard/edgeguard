"""
Evaluation script for EdgeGuard.
Used to get numbers for the paper (accuracy, F1, ECE, etc.)
Not super clean but it does the job.
"""

import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import confusion_matrix, f1_score
from src.model import EdgeGuardFinal
from src.metrics import compute_ece  # this is the TF version, but I also have a numpy one

# -------------------------------
# 1. Load model (hopefully it exists)
# -------------------------------
def load_model(model_path=None):
    # create model with default MC samples = 50
    model_obj = EdgeGuardFinal(mc_samples=50)
    if model_path:
        # try to load weights if provided
        model_obj.model.load_weights(model_path)
    else:
        print("Warning: no model path given, using untrained model (random weights). This will give garbage results.")
    return model_obj

# -------------------------------
# 2. Generate test data (replace with real data later)
# TODO: fix this when I have the real dataset
# -------------------------------
def get_test_data(num_samples=3000):
    radar = np.random.randn(num_samples, 100, 10).astype(np.float32)
    ais   = np.random.randn(num_samples, 50, 8).astype(np.float32)
    eoir  = np.random.randn(num_samples, 224, 224, 3).astype(np.float32)
    labels = np.random.randint(0, 3, size=(num_samples,))  # fake labels for now
    return [radar, ais, eoir], labels

# -------------------------------
# 3. Prediction with MC Dropout (T=50 usually)
# -------------------------------
def predict_mc(model, inputs, mc_samples=50):
    radar, ais, eoir = inputs
    mean, var, std = model.predict_with_uncertainty(
        radar, ais, eoir, mc_samples=mc_samples
    )
    return mean.numpy(), var.numpy(), std.numpy()

# -------------------------------
# 4. Main evaluation (spits out numbers)
# -------------------------------
def evaluate():
    print("Loading model... (might take a few seconds)")
    model = load_model()  # no pretrained weights here -> demo only
    
    print("Generating test data...")
    inputs, labels = get_test_data(3000)
    radar, ais, eoir = inputs
    
    print("Running inference with MC Dropout (T=50)...")
    start_time = time.time()
    probs, _, _ = predict_mc(model, inputs, mc_samples=50)
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"Latency (approx) for one batch: {elapsed_ms:.1f} ms (should be around 185 on Jetson)")
    
    preds = np.argmax(probs, axis=1)
    
    # metrics
    acc = np.mean(preds == labels)
    f1 = f1_score(labels, preds, average='macro')
    ece_val = compute_ece(probs, labels).numpy()
    
    print("\n===== Results (from this run) =====")
    print(f"Accuracy:  {acc:.4f}  (paper says 0.960 ± 0.003)")
    print(f"Macro F1:  {f1:.4f}  (paper: 0.961 ± 0.003)")
    print(f"ECE:       {ece_val:.4f}  (paper: 0.041 ± 0.005)")
    print(f"Latency:   {elapsed_ms:.1f} ms (projected)")
    
    cm = confusion_matrix(labels, preds)
    print("\nConfusion matrix:")
    print(cm)
    return acc, f1, ece_val, cm

# -------------------------------
# 5. Quick test for adversarial attack (AIS spoofing)
# -------------------------------
def test_ais_spoofing():
    print("\n--- Testing robustness against AIS spoofing ---")
    model = load_model()
    inputs, labels = get_test_data(500)
    radar, ais, eoir = inputs
    
    # clean
    probs_clean, _, _ = predict_mc(model, inputs, mc_samples=50)
    clean_acc = np.mean(np.argmax(probs_clean, axis=1) == labels)
    
    # add noise to AIS
    ais_noisy = ais + np.random.normal(0, 5.0, size=ais.shape)
    noisy_inputs = [radar, ais_noisy, eoir]
    probs_noisy, _, _ = predict_mc(model, noisy_inputs, mc_samples=50)
    noisy_acc = np.mean(np.argmax(probs_noisy, axis=1) == labels)
    
    print(f"Clean accuracy: {clean_acc:.4f}")
    print(f"Under attack:   {noisy_acc:.4f}")
    print(f"Drop: {((clean_acc - noisy_acc)*100):.1f}%  (paper claims attention helps by 5.3%)")

if __name__ == "__main__":
    evaluate()
    test_ais_spoofing()
