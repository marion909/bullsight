# ML Model Finetuning Guide

BullSight bietet jetzt**direkte Model-Finetuning im UI** - ohne Kommandozeile!

## 🎯 Workflow: Live Model Improvement

### **Schritt 1: Darts werfen & Bilder sammeln**

1. Öffnen Sie BullSight und navigieren Sie zu **Calibration Screen**
2. Werfen Sie 1-3 Pfeile aufs Dartboard
3. Klicken Sie **"Capture Test Image"** mehrmals (50-100 Bilder)
4. Bilder werden automatisch in `training_data/finetuning_data/images/` gespeichert

### **Schritt 2: Mit LabelImg Annotieren**

Installieren Sie LabelImg zum Markieren der Dart-Positionen:

```bash
pip install labelimg
labelimg
```

**Prozess:**
1. Klick: "Open Dir" → `training_data/finetuning_data/images/train/`
2. Für jedes Dart-Bild:
   - Zeichnen Sie ein Rechteck um jeden Dart (Bounding Box)
   - Oder tippen Sie `dart` als Label
3. Format: **YOLO (.txt Dateien)** 
4. Speichern, weiterziehen

**Labels werden automatisch erstellt in:**
```
training_data/finetuning_data/
├── images/
│   └── train/
│       ├── dart_001.jpg
│       └── ...
└── labels/
    └── train/
        ├── dart_001.txt  ← Label file
        └── ...
```

### **Schritt 3: Im ML Demo trainieren**

1. BullSight öffnen → **"🤖 ML Detection Demo"** Button
2. Klick: **"📚 Finetune Model"** (neuer großer blauer Button)
3. Im Dialog:
   - **Epochs:** 30-50 (mehr = bessere Genauigkeit, länger)
   - **Batch Size:** 8-16 (kleiner = weniger RAM)
   - Klick: **"▶️ Start Training"**

4. Training-Log zeigt Fortschritt live
5. Nach Fertigstellung: **"✅ Training successful"**

### **Schritt 4: Neues Modell aktivieren**

Das trainierte Modell wird automatisch zu `models/deepdarts_finetuned.pt` gespeichert.

**Neustart:**
```bash
python -m src.main
```

Das ML Demo wird jetzt ein Custom-Modell laden (nicht das syntetische Standard-Modell) - **mit echten Dart-Erkennungen!**

## 📊 Training-Tipps

### Beste Ergebnisse:
- ✅ **50-100 Bilder** für grundlegende Verbesserung
- ✅ **200+ Bilder** für Production-Qualität
- ✅ **Verschiedene Positionen:** Oben, Mitte, Unten, Ecken
- ✅ **Verschiedene Distanzen:** Nah und Fern vom Dartboard
- ✅ **Unterschiedliche Beleuchtung:** Hell und Dunkel

### Training-Parameter:

| Setting | Klein | Mittel | Groß |
|---------|-------|--------|------|
| **Epochs** | 15-25 | 30-50 | 50-100 |
| **Batch Size** | 4-8 | 8-16 | 16-32 |
| **Zeit (CPU)** | 30min | 1-2h | 2-4h |
| **Datensatz** | 20-50 Bilder | 50-150 | 200-500 |

### Learning:
- 📈 **Mehr Epochs** = besser, aber braucht längertc -> versuchen Sie zuerst 30-50
- 📉 **Zu viele Epochen** = Overfitting (funktioniert nur auf Trainings-Bildern)
- 💾 **Early Stopping** bei 15 Epochen ohne Verbesserung

## 🛠️ Troubleshooting

### Training startet nicht
```
❌ Dataset not found: training_data/finetuning_data
```
**Lösung:** Sammeln Sie zuerst Bilder mit "Capture Test Image"

### Keine Bilder gefunden
```
❌ Found 0 training images
```
**Lösung:** 
- Bewegen Sie Bilder von `training_data/finetuning_data/images/` zu `training_data/finetuning_data/images/train/`
- Oder vollständiges Neustart von "Capture Test Image"

### Modell wird nicht benutzt nach Training
- ✅ Restart: `python -m src.main`
- ✅ UI zeigt: "Custom finetuned (trained with your darts)"

## 🚀 Advanced: Lokales Training über SSH

Für größere Projekte oder Remote-Training:

```bash
# Remote (z.B. Raspberry Pi mit GPU)
ssh bullsight@192.168.0.221

cd BullSight
python train_real_model.py --prepare --train --epochs 100

# Modell wird automatisch in models/deepdarts_real.pt gespeichert
```

## 📈 Performance Monitoring

**Erwartete Verbesserungen:**
- Start: ~0% Erkennungsrate (synthetisches Modell)
- Nach 30 Epochen: ~60-80% (je nach Datensatz)
- Nach 100 Epochen: ~85-95%

**Metriken im Log:**
- `mAP50` - Precision bei IoU=0.5 (Ziel: >0.85)
- `Precision` - Wie viele erkannten Objects sind echte Darts
- `Recall` - Wie viele echte Darts wurden erkannt

## 💡 Nächste Schritte

1. **Finetune das Modell regelmäßig** mit neuen Bildern
2. **Testen Sie regelmäßig** im ML Demo Mode
3. **Exportieren Sie beste Modelle** vor Raspberry Pi Deployment

---

**Hinweis:** Das Finetuning-Feature ist experimental. Bei Problemen öffnen Sie ein Issue mit Training-Log!
