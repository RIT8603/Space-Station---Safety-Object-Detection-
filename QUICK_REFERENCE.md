# 🚀 QUICK REFERENCE CARD

## 🎯 Launch Commands

```bash
# Admin Dashboard (Recommended)
python -m streamlit run admin_dashboard.py

# Or double-click
launch_dashboard.bat

# Improved Detection App
python -m streamlit run streamlit_app_improved.py

# Train Enhanced Model
python train_enhanced.py
```

---

## 🔧 Detection Settings Quick Guide

### Maximum Detection
```
Confidence: 0.10-0.15
IoU: 0.3-0.4
Max Detections: 500
Use when: Crowded scenes, many objects
```

### Balanced (Recommended)
```
Confidence: 0.15-0.20
IoU: 0.4-0.5
Max Detections: 300
Use when: General monitoring
```

### High Precision
```
Confidence: 0.25-0.35
IoU: 0.5-0.6
Max Detections: 100
Use when: Critical alerts only
```

---

## 🤖 Model Selection Guide

| Model | Speed | Accuracy | When to Use |
|-------|-------|----------|-------------|
| YOLOv8n | ⚡⚡⚡ | ⭐⭐ | Real-time webcam |
| YOLOv8s | ⚡⚡ | ⭐⭐⭐ | Balanced performance |
| YOLOv8m | ⚡ | ⭐⭐⭐⭐ | High accuracy needed |
| Enhanced | ⚡ | ⭐⭐⭐⭐⭐ | Championship mode |

---

## 🚨 Alert Severity Levels

| Severity | Confidence | Color | Action |
|----------|-----------|-------|--------|
| 🚨 Critical | >90% | Red | Immediate attention |
| ⚠️ Warning | 70-90% | Orange | Monitor closely |
| ℹ️ Info | <70% | Blue | Routine logging |

---

## 📊 Dashboard Tabs Overview

1. **📹 Live Detection** - Upload images/videos, webcam feed
2. **🚨 Alerts** - Manage and acknowledge alerts
3. **📊 Analytics** - Charts, trends, statistics
4. **📈 Performance** - FPS, processing time, resources
5. **💾 Export Data** - CSV downloads, reports

---

## 🎨 7 Safety Classes

| # | Class | Color | Emoji |
|---|-------|-------|-------|
| 0 | OxygenTank | Cyan | 🔵 |
| 1 | NitrogenTank | Magenta | 🟣 |
| 2 | FirstAidBox | Green | 🟢 |
| 3 | FireAlarm | Red | 🔴 |
| 4 | SafetySwitchPanel | Orange | 🟠 |
| 5 | EmergencyPhone | Yellow | 🟡 |
| 6 | FireExtinguisher | Orange Red | 🔥 |

---

## 🐛 Quick Fixes

### Dashboard won't start
```bash
pip install streamlit plotly pandas ultralytics opencv-python
```

### No model loaded
```
1. Sidebar → Select Model
2. Click "Load Model" button
3. Wait for success message
```

### Webcam not working
```
1. Try camera index: 0, 1, or 2
2. Close Zoom, Teams, other webcam apps
3. Check Windows Privacy → Camera
```

### Database issues
```bash
del detection_logs.db
# Restart dashboard
```

---

## 📈 Performance Benchmarks

```
YOLOv8n:  ~100 FPS (GPU) | ~30 FPS (CPU) | 68-72% mAP
YOLOv8m:  ~50 FPS (GPU)  | ~10 FPS (CPU) | 75-80% mAP
Enhanced: ~40 FPS (GPU)  | ~8 FPS (CPU)  | 80%+ mAP
```

---

## 💡 Pro Tips

✅ **Start with YOLOv8n** for testing  
✅ **Use confidence 0.15** for balanced results  
✅ **Enable webcam** for live monitoring  
✅ **Check alerts tab** regularly  
✅ **Export data weekly** for reports  
✅ **Clear old data** monthly (30+ days)  
✅ **Load model first** before detection  
✅ **Lower confidence** for more detections  

---

## 📁 Key Files

```
admin_dashboard.py      - Main dashboard
database.py            - Database handler
train_enhanced.py      - Train new model
streamlit_app_improved.py - Alternative app
detection_logs.db      - SQLite database (auto-created)
best_enhanced.pt       - Enhanced model (after training)
```

---

## 🎯 Typical Workflow

### Quick Test
1. `launch_dashboard.bat`
2. Load YOLOv8n model
3. Upload test image
4. View results

### Real Monitoring
1. `launch_dashboard.bat`
2. Load enhanced model
3. Start webcam feed
4. Monitor alerts tab
5. Acknowledge critical alerts

### Training Session
1. `python train_enhanced.py`
2. Wait 4-8 hours
3. Model saved to `best_enhanced.pt`
4. Load in dashboard
5. Compare performance

---

## 🔐 Access URLs

- **Local**: http://localhost:8501
- **Network**: http://192.168.0.112:8501

---

## 📞 Need Help?

**Check these files:**
- `ADMIN_DASHBOARD_README.md` - Full dashboard guide
- `PROJECT_SUMMARY.md` - Complete overview
- `IMPROVEMENTS.md` - Detection tips
- `QUICKSTART.md` - Step-by-step guide

**Common Docs:**
```
CTRL+F to search within any .md file
```

---

## 🏆 Success Checklist

Before demo/submission:
- [ ] Dashboard launches successfully
- [ ] Model loads without errors
- [ ] Test image shows detections
- [ ] Webcam works (if applicable)
- [ ] Alerts generate properly
- [ ] Analytics show data
- [ ] Export CSV works
- [ ] All 7 classes detected

---

**Version**: 2.0 Enhanced  
**Last Updated**: 2025-01-10  
**Status**: ✅ Production Ready
