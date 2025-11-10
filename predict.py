"""
Space Station Challenge - Comprehensive Evaluation & Visualization
Generates:
- mAP@0.5, mAP@0.5:0.95
- Per-class AP table
- Confusion matrix (normalized + raw)
- Precision-Recall curves
- Top 50 failure cases
- Before/after augmentation comparison
"""

import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd

# Classes
CLASSES = ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 
           'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']

def evaluate_model(model_path, data_yaml, output_dir='runs/detect/evaluation'):
    """Run comprehensive evaluation"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    print()
    print(f"Model: {model_path}")
    print(f"Data:  {data_yaml}")
    print(f"Output: {output_dir}")
    print()
    
    # Load model
    print("📦 Loading model...")
    model = YOLO(model_path)
    
    # Run validation
    print("🔍 Running validation...")
    print()
    
    results = model.val(
        data=data_yaml,
        split='test',
        imgsz=1280,
        batch=16,
        conf=0.001,       # Low confidence for comprehensive evaluation
        iou=0.6,          # NMS IoU threshold
        max_det=300,      # Max detections per image
        plots=True,
        save_json=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True
    )
    
    print()
    print("=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print()
    
    # Extract metrics
    metrics = results.results_dict
    
    # Overall metrics
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    
    print(f"mAP@0.5:      {map50:.4f} ({map50*100:.2f}%)")
    print(f"mAP@0.5:0.95: {map50_95:.4f} ({map50_95*100:.2f}%)")
    print(f"Precision:    {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:       {recall:.4f} ({recall*100:.2f}%)")
    print()
    
    # Per-class metrics
    print("=" * 80)
    print("PER-CLASS AVERAGE PRECISION")
    print("=" * 80)
    print()
    
    # Create per-class table
    if hasattr(results, 'ap_class_index') and hasattr(results, 'ap'):
        class_metrics = []
        
        for idx, class_name in enumerate(CLASSES):
            ap50 = results.ap50[idx] if idx < len(results.ap50) else 0
            ap = results.ap[idx] if idx < len(results.ap) else 0
            
            class_metrics.append({
                'Class': class_name,
                'AP@0.5': f'{ap50:.4f}',
                'AP@0.5:0.95': f'{ap:.4f}',
                'AP@0.5 (%)': f'{ap50*100:.2f}%',
                'AP@0.5:0.95 (%)': f'{ap*100:.2f}%'
            })
        
        df = pd.DataFrame(class_metrics)
        print(df.to_string(index=False))
        print()
        
        # Save to CSV
        df.to_csv(output_dir / 'per_class_metrics.csv', index=False)
        print(f"✅ Saved per-class metrics: {output_dir / 'per_class_metrics.csv'}")
        print()
    
    # Generate visualizations
    generate_visualizations(results, output_dir)
    
    return results, map50, map50_95

def generate_visualizations(results, output_dir):
    """Generate all visualization plots"""
    
    print("=" * 80)
    print("📈 GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    output_dir = Path(output_dir)
    
    # 1. Confusion Matrix
    print("  Generating confusion matrix...")
    
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        cm = results.confusion_matrix.matrix
        
        # Normalized confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                   xticklabels=CLASSES + ['Background'],
                   yticklabels=CLASSES + ['Background'],
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix - Raw Counts', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Normalized
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
                   xticklabels=CLASSES + ['Background'],
                   yticklabels=CLASSES + ['Background'],
                   ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'Ratio'})
        axes[1].set_title('Confusion Matrix - Normalized', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ confusion_matrix.png")
    
    # 2. Precision-Recall Curve
    print("  Generating precision-recall curves...")
    
    if hasattr(results, 'curves'):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for idx, class_name in enumerate(CLASSES):
            if hasattr(results, 'curves_results') and idx in results.curves_results:
                px, py = results.curves_results[idx]['pr']
                ax.plot(px, py, linewidth=2, label=f'{class_name}')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves by Class', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ pr_curve.png")
    
    # 3. F1-Confidence Curve
    print("  Generating F1-confidence curve...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if hasattr(results, 'curves_results'):
        for idx, class_name in enumerate(CLASSES):
            if idx in results.curves_results:
                conf = results.curves_results[idx].get('conf', [])
                f1 = results.curves_results[idx].get('f1', [])
                if len(conf) > 0 and len(f1) > 0:
                    ax.plot(conf, f1, linewidth=2, label=f'{class_name}')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1-Confidence Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_curve.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ f1_curve.png")
    
    print()

def analyze_failures(model_path, data_yaml, output_dir='runs/detect/failures', num_failures=50):
    """Analyze and save top failure cases"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"🔍 ANALYZING TOP {num_failures} FAILURE CASES")
    print("=" * 80)
    print()
    
    # Load model
    model = YOLO(model_path)
    
    # Load data config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get test images
    test_path = Path(data_config['path']) / data_config['test']
    test_images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
    
    print(f"Found {len(test_images)} test images")
    print()
    
    # Run predictions
    print("Running predictions on test set...")
    
    failure_scores = []
    
    for img_path in tqdm(test_images):
        # Predict
        results = model.predict(img_path, imgsz=1280, conf=0.25, verbose=False)
        
        # Load ground truth
        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        
        if not label_path.exists():
            continue
        
        # Count ground truth objects
        with open(label_path, 'r') as f:
            gt_count = len(f.readlines())
        
        # Count predictions
        pred_count = len(results[0].boxes) if results and len(results) > 0 else 0
        
        # Calculate failure score (difference in detection count + low confidence)
        failure_score = abs(gt_count - pred_count)
        
        if pred_count > 0:
            avg_conf = float(results[0].boxes.conf.mean())
            failure_score += (1 - avg_conf)
        else:
            failure_score += 2  # High penalty for no detections
        
        failure_scores.append((img_path, failure_score, gt_count, pred_count))
    
    # Sort by failure score
    failure_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Save top failures
    print(f"\nSaving top {num_failures} failure cases...")
    
    for idx, (img_path, score, gt_count, pred_count) in enumerate(failure_scores[:num_failures]):
        # Read image
        image = cv2.imread(str(img_path))
        
        # Run prediction with visualization
        results = model.predict(img_path, imgsz=1280, conf=0.25, verbose=False)
        
        # Draw predictions
        if results and len(results) > 0:
            annotated = results[0].plot()
        else:
            annotated = image.copy()
        
        # Add failure info
        info_text = f"Failure Score: {score:.2f} | GT: {gt_count} | Pred: {pred_count}"
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save
        output_path = output_dir / f"failure_{idx+1:03d}_score{score:.2f}.jpg"
        cv2.imwrite(str(output_path), annotated)
    
    print(f"✅ Saved {num_failures} failure cases to: {output_dir}")
    print()
    
    # Print top 10 failure summary
    print("TOP 10 FAILURE CASES:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Score':<10} {'GT':<6} {'Pred':<6} {'Image'}")
    print("-" * 80)
    
    for idx, (img_path, score, gt_count, pred_count) in enumerate(failure_scores[:10]):
        print(f"{idx+1:<6} {score:<10.2f} {gt_count:<6} {pred_count:<6} {img_path.name}")
    
    print()

def compare_augmentation(dataset_root, augmented_root, num_samples=10):
    """Compare original vs augmented images side-by-side"""
    
    print("=" * 80)
    print("🖼️  AUGMENTATION COMPARISON")
    print("=" * 80)
    print()
    
    orig_path = Path(dataset_root) / 'train' / 'images'
    aug_path = Path(augmented_root) / 'train' / 'images'
    
    if not orig_path.exists() or not aug_path.exists():
        print("⚠️  Original or augmented dataset not found")
        return
    
    # Get sample images
    orig_images = list(orig_path.glob('*.jpg'))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, orig_img_path in enumerate(orig_images):
        # Original
        orig_img = cv2.imread(str(orig_img_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f'Original: {orig_img_path.name}', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Augmented versions
        aug_imgs = list(aug_path.glob(f'{orig_img_path.stem}_aug*'))
        
        for aug_idx in range(2):
            if aug_idx < len(aug_imgs):
                aug_img = cv2.imread(str(aug_imgs[aug_idx]))
                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                axes[idx, aug_idx + 1].imshow(aug_img)
                axes[idx, aug_idx + 1].set_title(f'Augmented {aug_idx+1}', fontsize=10)
            axes[idx, aug_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison: augmentation_comparison.png")
    print()

def main():
    # Configuration
    MODEL_PATH = 'runs/winning/yolov8x_champion/weights/best.pt'
    DATA_YAML = 'data_augmented.yaml'
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Please train the model first using train_winning.py")
        return
    
    # 1. Comprehensive evaluation
    results, map50, map50_95 = evaluate_model(MODEL_PATH, DATA_YAML)
    
    # 2. Failure analysis
    analyze_failures(MODEL_PATH, DATA_YAML, num_failures=50)
    
    # 3. Augmentation comparison
    compare_augmentation('dataset', 'dataset_aug', num_samples=10)
    
    print("=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - runs/detect/evaluation/confusion_matrix.png")
    print("  - runs/detect/evaluation/pr_curve.png")
    print("  - runs/detect/evaluation/f1_curve.png")
    print("  - runs/detect/evaluation/per_class_metrics.csv")
    print("  - runs/detect/failures/ (50 failure cases)")
    print("  - augmentation_comparison.png")
    print()
    print(f"Final Score: mAP@0.5 = {map50*100:.2f}%")
    
    if map50 >= 0.78:
        print("🏆 CHAMPIONSHIP TARGET ACHIEVED! (≥78%)")
    else:
        print(f"⚠️  Need {(0.78 - map50)*100:.2f}% more to reach 78% target")
    
    print()

if __name__ == '__main__':
    main()
