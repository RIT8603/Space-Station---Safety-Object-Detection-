@echo off
echo ============================================
echo Space Station Safety Detection - Environment Setup
echo ============================================

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

echo Creating conda environment 'EDU'...
call conda create -n EDU python=3.10 -y

echo Activating environment...
call conda activate EDU

echo Installing PyTorch with CUDA 12.1...
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo Installing Ultralytics YOLOv8...
pip install ultralytics==8.2.0

echo Installing core dependencies...
pip install opencv-python==4.10.0.84
pip install opencv-contrib-python==4.10.0.84
pip install albumentations==1.4.3
pip install wandb==0.16.6
pip install matplotlib==3.8.4
pip install seaborn==0.13.2
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install Pillow==10.3.0
pip install scikit-learn==1.5.0
pip install tqdm==4.66.4
pip install pyyaml==6.0.1
pip install tensorboard==2.16.2

echo Installing Streamlit for web app...
pip install streamlit==1.35.0
pip install streamlit-webrtc==0.47.1

echo Installing additional tools...
pip install markdown==3.6
pip install python-markdown-math==0.8
pip install reportlab==4.2.0
pip install fpdf2==2.7.8

echo Verifying PyTorch CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo To activate environment, run:
echo     conda activate EDU
echo.
pause
