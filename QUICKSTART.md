# Quick Start Guide - Space Station Challenge

## Prerequisites Checklist

Before running the project, you need:

1. ✅ **Python 3.10-3.11** (You have 3.13.9 - may have compatibility issues with PyTorch)
2. ❌ **Dataset** - You need to download the Falcon synthetic dataset
3. ❌ **Environment setup** - Need to install dependencies
4. ❓ **GPU** (Optional but recommended) - CUDA-capable GPU

---

## CRITICAL: Dataset Required!

**⚠️ You MUST have the dataset before running any scripts!**

The dataset should be structured like this:
```
D:\Features College\Hackathon\Space Station Challenge\
└── dataset/
    ├── train/
    │   ├── images/
    │   │   ├── img_001.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── img_001.txt
    │       └── ...
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

**Where to get the dataset:**
- Download from the Duality AI Falcon Challenge portal
- Or request access from the challenge organizers
- Place it in a folder called `dataset` in the current directory

---

## Step 1: Setup Environment (Alternative to Conda)

Since you don't have conda, use Python venv:

```powershell
# Create virtual environment
python -m venv venv_space_station

# Activate it
.\venv_space_station\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

---

## Step 2: Install Dependencies

**Option A: Quick Install (if you have CUDA GPU)**
```powershell
# Install PyTorch with CUDA
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

**Option B: CPU-Only Install (slower, but works)**
```powershell
# Install PyTorch CPU version
pip install torch torchvision

# Install other dependencies
pip install ultralytics opencv-python albumentations wandb matplotlib seaborn pandas numpy Pillow scikit-learn tqdm pyyaml tensorboard streamlit
```

---

## Step 3: Create Dataset Config

You need to create a `data.yaml` file pointing to your dataset:

```powershell
# Run this in PowerShell to create data.yaml
@"
path: D:\Features College\Hackathon\Space Station Challenge\dataset
train: train/images
val: valid/images
test: test/images

nc: 7
names:
  - OxygenTank
  - NitrogenTank
  - FirstAidBox
  - FireAlarm
  - SafetySwitchPanel
  - EmergencyPhone
  - FireExtinguisher
"@ | Out-File -FilePath "data.yaml" -Encoding utf8
```

---

## Step 4: Run the Project (Ordered Steps)

### 4.1 Explore Dataset (2 minutes)
```powershell
python data_explore.py
```
**Output:** Creates `class_distribution.png` and `sample_images.png`

### 4.2 Augment Dataset (30-60 minutes)
```powershell
python augment_dataset.py
```
**Output:** Creates `dataset_aug/` folder with 3x dataset size

### 4.3 Train Baseline Model (8-12 hours with GPU, ~1 week on CPU)
```powershell
python train_baseline.py
```
**Output:** Model saved to `runs/baseline/yolov8n_baseline/weights/best.pt`

### 4.4 Train Winning Model (48 hours with GPU, ~2 weeks on CPU)
```powershell
python train_winning.py
```
**Output:** Model saved to `runs/winning/yolov8x_champion/weights/best.pt`

⚠️ **Important:** During training, you'll be asked "Enable WandB logging? (y/n)". Type `n` if you don't have a WandB account.

### 4.5 Evaluate Model (10 minutes)
```powershell
python predict.py
```
**Output:** 
- `runs/detect/evaluation/` with metrics and plots
- `runs/detect/failures/` with failure cases
- `augmentation_comparison.png`

### 4.6 Run Web App
```powershell
streamlit run streamlit_app.py
```
**Opens browser at:** http://localhost:8501

---

## Quick Demo (If You Don't Have Time for Full Training)

If you just want to see the app working without training:

```powershell
# Download a pretrained YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Run the app (it will use the pretrained model)
streamlit run streamlit_app.py
```

Then upload any image to test detection (won't be space station-specific though).

---

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution:** Install PyTorch first
```powershell
pip install torch torchvision
```

### Issue: "Dataset not found"
**Solution:** Make sure you have the `dataset/` folder with train/valid/test splits

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use a smaller model:
- Edit `train_winning.py` and change batch size to 8 or 16
- Or use YOLOv8m instead of YOLOv8x

### Issue: Training is too slow on CPU
**Solution:** 
- Use Google Colab (free GPU): https://colab.research.google.com/
- Or use smaller model (YOLOv8n) and fewer epochs (50 instead of 300)

---

## Expected Timeline

| Task | GPU Time | CPU Time |
|------|----------|----------|
| Setup | 10 min | 10 min |
| Data Exploration | 2 min | 2 min |
| Augmentation | 30 min | 60 min |
| Baseline Training | 8 hours | 7 days |
| Winning Training | 48 hours | 14 days |
| Evaluation | 10 min | 30 min |
| **Total** | **~57 hours** | **~22 days** |

---

## Next Steps After Running

1. Check your mAP@0.5 score in evaluation results
2. If ≥78%, you've achieved championship level! 🏆
3. Generate the PDF report for submission
4. Record a demo video of the Streamlit app
5. Submit all files according to challenge guidelines

---

## Files You'll Submit

- `best.pt` (trained model)
- `CHAMPIONSHIP_REPORT.pdf` (convert from .md)
- `FALCON_INTEGRATION_BONUS.pdf` (convert from .md)
- Demo video (screen recording)
- This code repository

---

## Need Help?

Check the detailed documentation:
- `CHAMPIONSHIP_REPORT.md` - Full technical report
- `FALCON_INTEGRATION_BONUS.md` - Bonus submission details
- Problem statement: `problem_statement.pdf`
