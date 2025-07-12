# Underwater Image Enhancement using Lightweight CNN

This project presents a lightweight convolutional neural network (CNN) for enhancing underwater images degraded due to light absorption and scattering. The model follows an encoder-decoder architecture inspired by U-Net and is optimized for real-time deployment on edge devices like the Raspberry Pi.

## 🚀 Features
- Lightweight encoder-decoder CNN with skip connections
- Enhances visibility, color balance, and contrast in underwater images
- Trained on the UIEB benchmark dataset
- Real-time inference support via TorchScript
- Suitable for Raspberry Pi and other low-power edge devices

## 🧠 Model Architecture
- **Encoder**: 4 Conv → ReLU → MaxPool blocks  
  Features extracted with increasing depth (32 → 64 → 128 → 256)
- **Decoder**: 4 Upsample → Concat → Conv → ReLU blocks  
  Skip connections from encoder layers
- **Final Layer**: 1×1 convolution + Sigmoid activation to produce RGB image output in [0, 1] range

## 🧪 Training Details
- **Dataset**: UIEB – Underwater Image Enhancement Benchmark Dataset  
  (Paired images: raw underwater and human-labeled reference)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Epochs**: 12  
- **Batch Size**: 8
- **Framework**: PyTorch (Google Colab)

## 📦 Deployment
- The trained model is exported to a `.pt` TorchScript file
- Can be directly loaded and run on Raspberry Pi or any PyTorch-supported embedded device

```python
import torch
from model import LightUnderwaterEnhancer

model = LightUnderwaterEnhancer()
model.load_state_dict(torch.load("light_underwater_enhancer_script.pt", map_location='cpu'))
model.eval()
```

## Reference 
Can refer to https://colab.research.google.com/drive/1z7xNcxaJbyXL06gvIyg4UdDqJyNbGm6c?usp=sharing
