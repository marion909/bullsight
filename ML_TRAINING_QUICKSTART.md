# ⚡ ML Training Quickstart (200 Fotos Ready!)

## Status: BEREIT ZUM TRAINIEREN ✅

Sie haben:
- **200-201 echte Dartboard-Fotos** ✅
- **Gute Varianz** (1, 2, 3 Darts pro Bild) ✅
- **Automatisches Capture-Tool** ✅
- **Annotation-Tool vorbereitet** ✅

---

## 🚀 3-Minuten Schnellstart

### 1️⃣ Darts beschriften (5-10 Minuten)
```bash
python quick_label_darts.py
```

**Bedienung:**
- 🖱️ **Click + Drag** = Rechteck um Dart zeichnen
- ⌨️ **SPACE** = Speichern & nächstes Bild
- ⌨️ **Z** = Letzten Dart löschen
- ⌨️ **C** = Alle Darts löschen  
- ⌨️ **Q** = Abbrechen

Ergebnis: 200 `.txt` Dateien in `training_data/finetuning_data/labels/train/`

### 2️⃣ BullSight starten
```bash
python -m src.main
```

### 3️⃣ **1 Klick Training**
1. Navigiere: **🤖 ML Detection Demo**
2. Klick: **📚 Finetune Model**
3. Stelle ein: **Epochs = 30** (oder 50 für besser)
4. Klick: **▶️ Start Training**

**Fertig!** ✨ App trainiert deine Darts jetzt.

---

## 📊 Was passiert beim Training?

```
📥 Loading 201 training images...
🔍 Found all labels ✅
⚙️ Initializing YOLOv8n for fine-tuning...

🚀 EPOCH 1/30
  └─ Loss: 0.523, Precision: 0.91, Recall: 0.88

🚀 EPOCH 2/30  
  └─ Loss: 0.412, Precision: 0.93, Recall: 0.91

... (weitere Epochs)

✅ TRAINING COMPLETE in 18 minutes
📦 Model saved: models/deepdarts_finetuned.pt
💡 Restart BullSight to load new model
```

**Zeitleiste bei CPU:**
- 30 Epochs: ~15-20 Minuten
- 50 Epochs: ~30-35 Minuten
- 100 Epochs: ~60+ Minuten

---

## 🎯 Erwartete Ergebnisse

| Phase | Accuracy | Status |
|-------|----------|--------|
| **Vorher** (Base Model) | ❌ 0% | Erkennt keine Darts |
| **Nach Training** | ✅ 85-92% | Funktioniert zuverlässig |

**Im ML Demo sehen Sie dann:**
```
🎯 Live Detection:
✓ Dart #1: confidence=0.94 → Segment 20, Triple
✓ Dart #2: confidence=0.91 → Segment 5, Double  
✓ Dart #3: confidence=0.88 → Bull (50 Points)
```

---

## 💾 Dateistruktur nach Annotation

```
training_data/finetuning_data/
├── images/train/          (201 JPG Fotos)
│   ├── dart_training_20260207_162931_104.jpg
│   ├── dart_training_20260207_163238_107.jpg
│   └── ... (199 weitere)
│
└── labels/train/          (201 YOLO Annotationen)
    ├── dart_training_20260207_162931_104.txt
    ├── dart_training_20260207_163238_107.txt
    └── ... (199 weitere)
```

**Beispiel `.txt` Format:**
```
0 0.425 0.367 0.089 0.112
0 0.612 0.445 0.075 0.098
```
(Dart-Klasse 0, normalisierte Bounding-Box-Koordinaten)

---

## 🎮 Nach dem Training

**Nach Neustart von BullSight:**
```python
# App lädt automatisch:
✨ Using custom finetuned model (201 images, 30 epochs)
```

**So überprüfen Sie Erfolg:**
1. Gehen Sie zu: **🤖 ML Detection Demo**
2. Halten Sie einen Dart ins Bild
3. ✅ Sie sollten **grüne Bounding Boxes** sehen  
4. ✅ **Hohe Confidence-Scores** (0.90+)

---

## ⚙️ Hyperparameter-Guide

### Standard (empfohlen für 200 Bilder)
```
Epochs: 30
Batch Size: 8
Learning Rate: 0.001 (auto)
```
→ **Zeit: 15-20 Min** | **Qualität: 85-90%**

### Hohe Qualität (wenn Zeit kein Problem)
```
Epochs: 50-100
Batch Size: 8
Learning Rate: 0.001
```
→ **Zeit: 30-60 Min** | **Qualität: 90-95%**

### Schnell (Test-Run)
```
Epochs: 10
Batch Size: 16
Learning Rate: 0.01
```
→ **Zeit: 5-8 Min** | **Qualität: 70-80%**

---

## 🐛 Troubleshooting

### ❌ "No labels found"
**Ursache:** `quick_label_darts.py` nicht ausgeführt  
**Lösung:**
```bash
python quick_label_darts.py
# Annotieren Sie ALLE 201 Bilder
```

### ❌ "Dataset images not found"
**Ursache:** Falsche `data.yaml` Pfade  
**Lösung:**
```bash
python setup_finetuning_dataset.py
```

### ❌ "CUDA out of memory"
**Ursache:** Batch Size zu groß  
**Lösung:** In UI auf **Batch Size: 4** reduzieren

### ❌ "Training nur 1% genau"
**Ursache:** Zu wenige Trainingsdaten oder falsche Annotationen  
**Lösung:**
- Sammeln Sie 50-100 weitere Bilder
- Überprüfen Sie Annotation-Qualität in `quick_label_darts.py`
- Trainieren Sie 100 statt 30 Epochs

---

## 📈 Iteratives Verbessern

**Erste Iteration (JETZT):**
```
200 Bilder → Training → 85% Accuracy
```

**Zweite Iteration (Optional):**
```
+ 100 neue Bilder schwieriger Fälle 
→ Training mit 300 Bildern → 92% Accuracy
```

**Tipps für schwierige Fälle:**
- Sehr dunkle Aufnahmen
- Darts von hinten (nur Körper sichtbar)
- Darts die sich überlappen
- Verschiedene Dart-Farben (rot, schwarz, gold)

---

## ✅ Checkliste vor Training

- [ ] **201 Trainings-Bilder** in `training_data/finetuning_data/images/train/`
- [ ] **quick_label_darts.py** läuft und erstellt Labels
- [ ] **Alle 201 Labels** in `training_data/finetuning_data/labels/train/`
- [ ] **BullSight startet** (`python -m src.main`)
- [ ] **ML Detection Demo** öffnet ohne Fehler
- [ ] **"📚 Finetune Model"** Button sichtbar

---

## 🎓 Was Sie jetzt können

Nach dem Training:
- ✅ **Live Dart-Erkennung** im ML Demo
- ✅ **Automatische Score-Berechnung** basierend auf erkannten Darts
- ✅ **Custom Model** optimiert für IHRE Dartboards
- ✅ **Iteratives Verbessern** durch neue Trainings-Runden

---

**Bereit?** Dann los! 🚀

```bash
# JETZT STARTEN:
python quick_label_darts.py
```

Fragen? Siehe [TRAINING_GUIDE.md](TRAINING_GUIDE.md) für Details.
