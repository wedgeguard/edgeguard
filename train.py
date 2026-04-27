"""
Training script for EdgeGuard model (the one from DTISD 2026 paper).
Not super polished, but it works :)
"""

import tensorflow as tf
import numpy as np
from src.model import EdgeGuardFinal

# -------------------------------
# 1. Training settings (I tried to follow the paper as much as possible)
# -------------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001   # same as 1e-4
NUM_RUNS = 10  # number of independent runs, they used this to compute mean/std in the paper

# -------------------------------
# 2. Dummy data generator (real data should be loaded here, but for demo it's random)
# -------------------------------
def generate_dummy_data(num_samples=1000):
    # TODO: replace with actual data loading
    radar = np.random.randn(num_samples, 100, 10).astype(np.float32)
    ais   = np.random.randn(num_samples, 50, 8).astype(np.float32)
    eoir  = np.random.randn(num_samples, 224, 224, 3).astype(np.float32)
    # random labels (0,1,2) - just placeholder
    labels = np.random.randint(0, 3, size=(num_samples,))
    return [radar, ais, eoir], labels

# -------------------------------
# 3. Build and compile model (maybe I should have put this inside the loop? but it's fine)
# -------------------------------
def build_and_compile_model():
    # create model with 50 MC samples (default)
    edgeguard_model = EdgeGuardFinal(mc_samples=50).model
    edgeguard_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return edgeguard_model

# -------------------------------
# 4. Training loop with multiple runs (as in paper)
# -------------------------------
def train():
    print("Starting EdgeGuard training pipeline (using simulated data for now).")
    print(f"Settings: batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LEARNING_RATE}, runs={NUM_RUNS}")

    accuracies = []  # to collect final validation accuracies from each run

    for run_id in range(1, NUM_RUNS+1):
        print(f"\n--- Run {run_id}/{NUM_RUNS} ---")
        # generate fresh data for each run (maybe not necessary but ok)
        (x_radar, x_ais, x_eoir), y = generate_dummy_data(num_samples=2000)

        model = build_and_compile_model()

        history = model.fit(
            [x_radar, x_ais, x_eoir], y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            verbose=1   # shows progress, I like to see what's happening
        )
        final_val_acc = history.history['val_accuracy'][-1]
        accuracies.append(final_val_acc)
        print(f"Run {run_id} done. Validation accuracy: {final_val_acc:.4f}")

        # save model checkpoint (optional)
        model.save(f"models/edgeguard_run{run_id}.h5")

    # -------------------------------
    # 5. Final stats (matching the paper's Table 4, I guess)
    # -------------------------------
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print("\n" + "="*50)
    print(f"Overall validation accuracy over {NUM_RUNS} runs: {mean_acc:.4f} ± {std_acc:.4f}")
    print("(Target from paper: 96.0% ± 0.3)")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)  # create folder if doesn't exist
    train()
