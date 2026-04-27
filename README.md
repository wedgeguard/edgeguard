# EdgeGuard: Hybrid Simulation-Real Framework for Maritime Threat Detection
assets/Architecture.ping
**DTISD 2026 – Best Paper Candidate**

## Overview
Official implementation of EdgeGuard, a real-time edge AI framework for maritime threat detection at the Bab el-Mandeb Strait.

## Key Features
| Feature | Implementation |
| :--- | :--- |
| Attention-based fusion | Scalar modality attention via trainable scores |
| Uncertainty quantification | MC Dropout with T = 50 (vectorized via `tf.tile`) |
| Confidence calibration | Expected Calibration Error (ECE) monitoring |
| Optimized backbone | cuDNN LSTMs + fine‑tuned MobileNetV2 |
| Prediction time | 185 ms on Jetson AGX Orin (projected) |

## Repository Structure
```
edgeguard/
├── README.md
├── requirements.txt
└── src/
    ├── model.py
    └── metrics.py
```

## Quick Start
```bash
git clone https://github.com/waleededgeguard/edgeguard.git
cd edgeguard
pip install -r requirements.txt
python -c "from src.model import EdgeGuardFinal; import numpy as np; \
model = EdgeGuardFinal(); \
radar = np.random.randn(1,100,10); ais = np.random.randn(1,50,8); \
eoir = np.random.randn(1,224,224,3); \
mean, var, std = model.predict_with_uncertainty(radar, ais, eoir); \
print('Prediction:', mean.numpy(), 'Uncertainty:', std.numpy())"
```

## Citation
If you use this code, please cite our DTISD 2026 paper.

## Contact
[waleedsaifmoqbelsaeed@gmail.coml
