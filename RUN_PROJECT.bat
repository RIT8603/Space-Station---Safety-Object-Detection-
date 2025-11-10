@echo off
echo ============================================
echo Space Station Challenge - Quick Start
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "venv_space_station\" (
    echo Creating virtual environment...
    python -m venv venv_space_station
    echo.
)

echo Activating virtual environment...
call venv_space_station\Scripts\activate.bat

echo.
echo ============================================
echo What would you like to do?
echo ============================================
echo.
echo 1. Install dependencies (first time setup)
echo 2. Explore dataset (2 minutes)
echo 3. Augment dataset (30-60 minutes)
echo 4. Train baseline model (8+ hours)
echo 5. Train winning model (48+ hours)
echo 6. Evaluate model
echo 7. Run Streamlit web app
echo 8. Quick demo (pretrained model)
echo 9. Exit
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto explore
if "%choice%"=="3" goto augment
if "%choice%"=="4" goto train_baseline
if "%choice%"=="5" goto train_winning
if "%choice%"=="6" goto evaluate
if "%choice%"=="7" goto webapp
if "%choice%"=="8" goto demo
if "%choice%"=="9" goto end

:install
echo.
echo ============================================
echo Installing Dependencies
echo ============================================
echo.
echo This will take 5-10 minutes...
echo.
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python albumentations wandb matplotlib seaborn pandas numpy Pillow scikit-learn tqdm pyyaml tensorboard streamlit
echo.
echo ✅ Installation complete!
echo.
pause
goto end

:explore
echo.
echo ============================================
echo Exploring Dataset
echo ============================================
echo.
python data_explore.py
echo.
echo ✅ Check class_distribution.png and sample_images.png
echo.
pause
goto end

:augment
echo.
echo ============================================
echo Augmenting Dataset (3x size)
echo ============================================
echo.
echo This will take 30-60 minutes...
echo.
python augment_dataset.py
echo.
echo ✅ Augmented dataset saved to dataset_aug/
echo.
pause
goto end

:train_baseline
echo.
echo ============================================
echo Training Baseline Model
echo ============================================
echo.
echo WARNING: This will take 8+ hours with GPU!
echo.
set /p confirm="Continue? (y/n): "
if /i "%confirm%"=="y" (
    python train_baseline.py
    echo.
    echo ✅ Model saved to runs/baseline/yolov8n_baseline/weights/best.pt
)
echo.
pause
goto end

:train_winning
echo.
echo ============================================
echo Training Winning Model
echo ============================================
echo.
echo WARNING: This will take 48+ hours with GPU!
echo.
set /p confirm="Continue? (y/n): "
if /i "%confirm%"=="y" (
    python train_winning.py
    echo.
    echo ✅ Model saved to runs/winning/yolov8x_champion/weights/best.pt
)
echo.
pause
goto end

:evaluate
echo.
echo ============================================
echo Evaluating Model
echo ============================================
echo.
python predict.py
echo.
echo ✅ Check runs/detect/evaluation/ for results
echo.
pause
goto end

:webapp
echo.
echo ============================================
echo Starting Streamlit Web App
echo ============================================
echo.
echo Opening browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run streamlit_app.py
goto end

:demo
echo.
echo ============================================
echo Quick Demo (Pretrained Model)
echo ============================================
echo.
echo Downloading pretrained YOLOv8 model...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
echo.
echo Starting web app with pretrained model...
echo.
streamlit run streamlit_app.py
goto end

:end
echo.
echo ============================================
echo Done! Check QUICKSTART.md for more details
echo ============================================
pause
