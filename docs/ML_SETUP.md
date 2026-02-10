# BullSight ML Dart Detection Setup

## Überblick

BullSight kann ML-basierte Dart-Erkennung mit YOLOv8 verwenden. Diese ist wesentlich robuster als klassische Computer Vision bei:
- Unterschiedlichen Lichtverhältnissen
- Verschiedenen Dart-Typen und -Farben
- Schrägen Kamerawinkeln
- Teilverdeckungen

## 🎯 ML Demo Modus

BullSight hat einen integrierten **ML Demo Modus** zum Testen und Visualisieren der Dart-Erkennung!

### Demo Modus starten:

1. Öffne BullSight
2. Klicke auf **"🤖 ML Detection Demo"** im Hauptmenü
3. Wähle zwischen:
   - **Live Camera**: Echtzeit-Erkennung vom Kamera-Feed
   - **Test Image**: Lade gespeicherte Bilder zum Testen

### Features des Demo Modus:

- ✅ **Live-Visualisierung**: Sieh die ML-Erkennung in Echtzeit
- ✅ **Bounding Boxes**: Grüne/gelbe/orange Boxen je nach Confidence
- ✅ **Confidence Scores**: Prozent-Anzeige für jede Erkennung
- ✅ **Positionsanzeige**: Pixel-Koordinaten und Dartboard-Feld
- ✅ **Confidence Threshold**: Slider zum Anpassen (10-95%)
- ✅ **Board Overlay**: Zeige Kalibrierungsringe an
- ✅ **Multi-Dart**: Erkennt mehrere Darts gleichzeitig

### Farb-Kodierung:

- 🟢 **Grün**: Hohe Confidence (>70%)
- 🟡 **Gelb**: Mittlere Confidence (50-70%)
- 🟠 **Orange**: Niedrige Confidence (<50%)

## Installation

### 1. ML-Abhängigkeiten installieren

```bash
pip install -r requirements-ml.txt
```

Dies installiert:
- `ultralytics` (YOLOv8)
- `torch` (PyTorch)
- `torchvision`

### 2. Model Options

#### Option A: Schnellstart mit vortrainiertem Modell (Empfohlen)

Das YOLOv8-nano Basismodell wird automatisch heruntergeladen. Es funktioniert bereits für Objekt-Erkennung, braucht aber Fine-Tuning für optimale Dart-Detection.

**Aktivieren:**
```python
# In src/vision/dart_detector.py __init__:
detector = DartDetector(use_ml=True)
```

#### Option B: Eigenes Modell trainieren (Beste Genauigkeit)

1. **Bilder sammeln** (mindestens 50-100):
   - Verwende die "Capture Test Image" Funktion im Kalibrierungsscreen
   - Verschiedene Dart-Positionen auf der Scheibe
   - Verschiedene Lichtverhältnisse
   - Speicherort: `test_images/`

2. **Bilder annotieren**:
   - Verwende [LabelImg](https://github.com/heartexlabs/labelImg) oder [Roboflow](https://roboflow.com)
   - Markiere die Dart-Spitze mit einer Bounding Box
   - Label: "dart"
   - Export im YOLO-Format

3. **Dataset strukturieren**:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── data.yaml
   ```

4. **data.yaml erstellen**:
   ```yaml
   path: /pfad/zu/dataset
   train: images/train
   val: images/val
   
   nc: 1  # Number of classes
   names: ['dart']
   ```

5. **Modell trainieren**:
   ```python
   from src.vision.ml_dart_detector import train_model
   
   train_model(
       data_yaml='dataset/data.yaml',
       epochs=100,
       model_size='n'  # n=nano, s=small, m=medium
   )
   ```

6. **Trainiertes Modell verwenden**:
   ```python
   detector = DartDetector(
       use_ml=True,
       ml_model_path='bullsight_training/dart_detector/weights/best.pt'
   )
   ```

## Aktivierung in BullSight

### Automatisch (Empfohlen)

ML wird **automatisch aktiviert** wenn Ultralytics installiert ist! Einfach:

```bash
pip install ultralytics torch torchvision
python -m src.main
```

Das war's! BullSight erkennt automatisch dass ML verfügbar ist.

### Manuell (Optional)

Falls du ML manuell steuern möchtest:

```bash
# ML explizit aktivieren
$env:BULLSIGHT_USE_ML=1
python -m src.main

# ML explizit deaktivieren (auch wenn installiert)
$env:BULLSIGHT_USE_ML=0
python -m src.main

# Eigenes Modell verwenden
$env:BULLSIGHT_ML_MODEL="pfad/zu/model.pt"
$env:BULLSIGHT_ML_CONFIDENCE=0.6
python -m src.main
```

### Via Code

In `src/main.py`:

```python
# ML Detection aktivieren
self.detector = DartDetector(
    use_ml=True,  # ML aktivieren
    ml_model_path='models/dart_detector.pt',  # Optional: eigenes Modell
    ml_confidence=0.5  # Mindest-Konfidenz (0.0-1.0)
)
```

## Performance

### Raspberry Pi 4

- **YOLOv8-nano**: ~100ms pro Frame (gut nutzbar)
- **YOLOv8-small**: ~250ms pro Frame (langsamer aber genauer)

### Desktop/Laptop

- **YOLOv8-nano**: ~20-50ms pro Frame
- **YOLOv8-small**: ~50-100ms pro Frame

## Troubleshooting

### ImportError: No module named 'ultralytics'

```bash
pip install ultralytics torch torchvision
```

### CUDA out of memory

Verwende kleineres Modell (nano statt small) oder reduziere Bildgröße.

### Schlechte Erkennungsrate

1. Sammle mehr Trainingsbilder
2. Achte auf Bildqualität (Beleuchtung, Schärfe)
3. Annotiere präzise (nur die Dart-Spitze)
4. Trainiere länger (mehr Epochs)

## Vergleich: Classical CV vs ML

| Aspekt | Classical CV | ML (YOLO) |
|--------|-------------|-----------|
| Setup  | Sofort bereit | Training nötig |
| Robustheit | Mittelmäßig | Sehr gut |
| Licht-Varianz | Empfindlich | Robust |
| Geschwindigkeit | Sehr schnell | Schnell |
| Winkel-Toleranz | Begrenzt | Sehr gut |
| Mehrfach-Darts | Mühsam | Einfach |

## Best Practices

1. **Start mit Classical CV**: Teste erst das bestehende System
2. **Sammle Daten während Nutzung**: Nutze "Capture Test Image"
3. **Iteratives Training**: Training → Test → Mehr Daten → Retraining
4. **Data Augmentation**: Roboflow bietet automatische Augmentation
5. **Hybrid-Ansatz**: ML als Primary, Classical CV als Fallback

## Nächste Schritte

1. ☐ ML-Abhängigkeiten installieren
2. ☐ 50-100 Dart-Bilder sammeln
3. ☐ Bilder mit LabelImg/Roboflow annotieren
4. ☐ Dataset vorbereiten (train/val split)
5. ☐ Modell trainieren (1-2 Stunden)
6. ☐ In BullSight aktivieren und testen
7. ☐ Bei Bedarf: mehr Daten sammeln und nachtrainieren
