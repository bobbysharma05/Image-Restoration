# IMAGE DEBLURRING USING DEEP LEARNING
### UNet + EfficientNet-B3 | PyTorch | GoPro Dataset

---

## 📌 Project Overview

This project implements an end-to-end deep learning pipeline for **blind image deblurring** — restoring sharp images from motion-blurred inputs without any prior knowledge of the blur kernel. The model is built using a **UNet architecture with an EfficientNet-B3 encoder** pretrained on ImageNet, trained on the GoPro Deblur Dataset.

> **Course:** CS Final Year Project — Deep Learning & Computer Vision  
> **Dataset:** GoPro Deblur Dataset (Kaggle)  
> **Framework:** PyTorch + Segmentation Models PyTorch (smp)

---

## 🏗️ Architecture

```
Input (Blurry Image 256×256)
        ↓
┌─────────────────────────────┐
│   EfficientNet-B3 Encoder   │  ← Pretrained on ImageNet
│   (Frozen for first 10 ep)  │
└────────────┬────────────────┘
             │  Skip Connections
             ↓
┌─────────────────────────────┐
│   UNet Decoder              │
│   + SCSE Attention          │  ← Spatial & Channel SE
│   + Dropout (0.2)           │  ← Regularization
└────────────┬────────────────┘
             ↓
Output (Sharp Image 256×256)
```

| Component | Detail |
|-----------|--------|
| Backbone | EfficientNet-B3 (ImageNet weights) |
| Framework | UNet via `segmentation_models_pytorch` |
| Decoder Attention | SCSE (Spatial + Channel Squeeze & Excitation) |
| Decoder Dropout | 0.2 |
| Total Parameters | 13,222,985 |
| Input Resolution | 256 × 256 |

---

## 📂 Dataset Structure

```
/content/dataset/gopro_deblur/
├── blur/
│   └── images/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
└── sharp/
    └── images/
        ├── image_001.png
        ├── image_002.png
        └── ...
```

- **Total pairs:** 1,029 aligned blur ↔ sharp image pairs
- **Train / Val split:** 90% / 10% (926 train, 103 val)
- **Source:** [GoPro Deblur Dataset on Kaggle](https://www.kaggle.com/datasets/rahulbhalley/gopro-deblur)

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/image-deblurring.git
cd image-deblurring

# Install dependencies
pip install torch torchvision
pip install segmentation_models_pytorch
pip install pytorch-msssim
pip install albumentations
pip install scipy pillow matplotlib tqdm numpy
```

Or if running on **Google Colab**:

```python
!pip install segmentation_models_pytorch pytorch-msssim albumentations -q
```

---

## 🚀 Training

> **Recommended: Run on Google Colab with GPU runtime**  
> Runtime → Change runtime type → T4 GPU

### 1. Open the Notebook
Upload `main.ipynb` to [Google Colab](https://colab.research.google.com) and enable GPU.

### 2. Download the Dataset
```python
# Upload your kaggle.json when prompted
from google.colab import files
uploaded = files.upload()  # upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d rahulbhalley/gopro-deblur
!unzip -q gopro-deblur.zip -d /content/dataset
```

### 3. Install Dependencies
```python
!pip install segmentation_models_pytorch pytorch-msssim albumentations -q
```

### 4. Configure Training
```python
DATASET_ROOT = '/content/dataset/gopro_deblur'
BATCH_SIZE   = 8
IMG_SIZE     = 256
EPOCHS       = 100
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH    = 'deblur_best.pth'
```

### 5. Run All Cells
Training will:
- Freeze the encoder for the first 10 epochs
- Unfreeze and fine-tune the full network from epoch 11
- Save the best model checkpoint automatically to `deblur_best.pth`
- Apply early stopping with patience of 10 epochs
- Visualize input → output → target every 5 epochs

---

## 📊 Training Results

### Final Run (100 Epoch Config)

| Metric | Value |
|--------|-------|
| Best Val Loss | **0.1026** |
| Early Stopped At | Epoch 38 / 100 |
| Train Loss (final) | ~0.104 |
| Val Loss (best) | 0.1026 |
| PSNR (approx.) | ~30.2 dB |
| SSIM (approx.) | ~0.89 |
| Training Time | ~1.25 hours (NVIDIA GPU) |

### Overfitting Resolution

One of the core challenges faced and resolved in this project:

| Problem | Overfitting — val loss stuck at 0.15–0.17 while train kept dropping |
|---------|---------------------------------------------------------------------|
| Root Cause | No decoder regularization, weak augmentation, unstable early training |

**Fixes Applied:**

| Fix | Impact |
|-----|--------|
| Decoder Dropout (0.2) | Most impactful — forces decoder to generalize |
| AdamW + weight_decay=1e-4 | Penalizes large weights implicitly |
| Encoder freezing (first 10 epochs) | Stabilizes early training |
| Stronger augmentation pipeline | More diverse training distribution |
| CosineAnnealingLR | Smoother, more principled LR decay |
| Early stopping (patience=10) | Prevents training past convergence |

---

## 🔢 Loss Function — BetterLoss

A composite loss combining three complementary objectives:

```
Loss = 0.5 × L1 + 0.4 × (1 - SSIM) + 0.1 × VGG_Perceptual
```

| Component | Weight | What it optimizes |
|-----------|--------|-------------------|
| L1 Loss | 0.5 | Pixel-level accuracy |
| SSIM Loss | 0.4 | Structural & texture similarity |
| VGG Perceptual Loss | 0.1 | High-level perceptual realism |

---

## 🔄 Data Augmentation Pipeline

Applied during training to prevent overfitting and improve generalization:

```
✅ Horizontal flip          → 50% probability
✅ Vertical flip            → 50% probability
✅ Random rotation          → 90° / 180° / 270°, 50% probability
✅ Random crop + resize     → 75–100% of original, 60% probability
✅ Gaussian blur synthesis  → radius 1.5–4.0, synthetic blur
✅ Motion blur synthesis    → kernel 9/13/17/21px, horizontal or diagonal
```

Synthetic blur synthesis (60% chance per image) augments blur diversity beyond what the GoPro dataset alone provides.

---

## 🔬 Inference

### Run on a single blurry image:

```python
# Load trained model
model.load_state_dict(torch.load('deblur_best.pth', map_location=DEVICE))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
img = Image.open('your_blurry_image.jpg').convert('RGB')
inp = transform(img).unsqueeze(0).to(DEVICE)

# Inference
with torch.no_grad():
    output = model(inp)

# Save result
result = transforms.ToPILImage()(output.squeeze().cpu().clamp(0, 1))
result = result.resize(img.size, Image.LANCZOS)  # restore original resolution
result.save('deblurred_output.png')
```

### In Google Colab (interactive):
```python
predict()  # will prompt you to upload a blurry image
```

---

## 📁 File Structure

```
project/
├── main.ipynb     # Full training + inference pipeline (Google Colab)
├── README.md      # This file
└── Summary.pdf    # Project summary document
```

---

## 📦 Requirements

```
torch>=1.12.0
torchvision>=0.13.0
segmentation-models-pytorch>=0.3.0
pytorch-msssim>=0.2.1
albumentations>=1.3.0
scipy>=1.9.0
Pillow>=9.0.0
matplotlib>=3.5.0
tqdm>=4.64.0
numpy>=1.23.0
```

---

## 🔮 Future Work

- [ ] Replace UNet with **Restormer** or **NAFNet** (Transformer-based SOTA)
- [ ] Train at full resolution (1280×720) instead of 256×256
- [ ] Add **FFT frequency loss** to directly target missing high frequencies
- [ ] Experiment with **GAN-based training** for sharper perceptual outputs
- [ ] Train on **RealBlur dataset** for real-world camera shake generalization
- [ ] Implement **Test-Time Augmentation (TTA)** for inference quality boost
- [ ] Deploy as a **web application** using Gradio or Streamlit

---

## 📈 Key Observations

- Early stopping fired at **epoch 38**, not 100 — the model, not the epoch count, decides convergence
- Encoder freezing for 10 epochs was critical — unfreezing at epoch 11 immediately produced a new best val loss
- Val loss ≈ train loss throughout — confirms **zero overfitting** in final model
- Loss floor at ~0.10 is a natural limit given dataset size (1029 pairs) and training resolution (256×256)
15

---

*Built with PyTorch • EfficientNet-B3 • UNet • SCSE Attention • GoPro Dataset*