"""
Data loading utilities for EdgeGuard.
I need to load Singapore Maritime Dataset, MarineTraffic AIS, and simulated radar.
This is not perfect, but it's a start.
"""

import numpy as np
import os

# -------------------------------
# Load EO/IR images from Singapore dataset
# (assuming the dataset is organized in folders)
# -------------------------------
def load_singapore_data(data_dir):
    """
    Expected structure: data_dir/class_name/*.jpg
    Returns: images (list), labels (list)
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    class_map = {'Normal':0, 'Suspicious':1, 'Hostile':2}
    images = []
    labels = []
    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: {class_dir} not found, skipping")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg','.png','.jpeg')):
                img = load_img(os.path.join(class_dir, fname), target_size=(224,224))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels)

# -------------------------------
# Load AIS trajectories (CSV format)
# -------------------------------
def load_ais_data(csv_path, max_len=50):
    """
    Reads AIS data from CSV. Expected columns: timestamp, mmsi, lat, lon, sog, cog, etc.
    Returns: numpy array of shape (num_vessels, max_len, 8)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # TODO: implement actual preprocessing (group by MMSI, pad sequences)
    # For now, return random data as placeholder
    print("Warning: AIS loader not fully implemented, returning random data")
    num_vessels = 5000
    return np.random.randn(num_vessels, max_len, 8).astype(np.float32)

# -------------------------------
# Simulate radar data (or load from MATLAB .mat)
# -------------------------------
def load_radar_data(mat_path=None, num_frames=50000):
    """
    If mat_path is given, load from .mat file. Otherwise generate synthetic.
    Returns: numpy array of shape (num_frames, 100, 10)
    """
    if mat_path and os.path.exists(mat_path):
        try:
            import scipy.io as sio
            data = sio.loadmat(mat_path)
            # assume variable name 'radar_data'
            radar = data['radar_data'].astype(np.float32)
            print(f"Loaded radar data from {mat_path}, shape {radar.shape}")
            return radar
        except Exception as e:
            print(f"Failed to load {mat_path}: {e}")
    print("Generating synthetic radar data (calibrated to real stats)")
    # simulate with mean -15.2 dB, std 2.1 (as in paper table III)
    clutter = np.random.normal(-15.2, 2.1, size=(num_frames, 100, 10))
    # add signal part (simple model)
    signal = np.random.uniform(-5, 25, size=(num_frames, 100, 10))
    radar = clutter + signal
    return radar.astype(np.float32)

# -------------------------------
# Combine all modalities (align by time)
# I need to make sure they have the same number of samples
# -------------------------------
def preprocess_and_align(radar_data, ais_data, eoir_data):
    """
    Take three arrays and crop/pad to same length.
    Returns: aligned (radar, ais, eoir) and labels if provided
    """
    min_len = min(len(radar_data), len(ais_data), len(eoir_data))
    radar_aligned = radar_data[:min_len]
    ais_aligned = ais_data[:min_len]
    eoir_aligned = eoir_data[:min_len]
    return radar_aligned, ais_aligned, eoir_aligned

# -------------------------------
# Test the loaders (just for debugging)
# -------------------------------
if __name__ == "__main__":
    # quick test
    print("Testing data loader functions...")
    eoir_imgs, eoir_labels = load_singapore_data("data/EOIR")
    print(f"EO/IR images: {eoir_imgs.shape}, labels: {eoir_labels.shape}")
    
    ais = load_ais_data("data/AIS/sample.csv")
    print(f"AIS data shape: {ais.shape}")
    
    radar = load_radar_data("data/radar/simulated.mat")
    print(f"Radar data shape: {radar.shape}")
    
    print("Preprocessing test:")
    r, a, e = preprocess_and_align(radar, ais, eoir_imgs)
    print(f"Aligned shapes: radar {r.shape}, AIS {a.shape}, EOIR {e.shape}")
