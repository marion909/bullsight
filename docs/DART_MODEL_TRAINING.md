# Training Custom Dart Detection Models

## Quick Start

BullSight unterstützt benutzerdefinierte YOLOv8-Modelle für Dart-Erkennung. Folgen Sie dieser Anleitung, um ein trainiertes Modell zu erstellen.

## Option 1: Automatisches Training mit DartsVision-Dataset

Das DartsVision-Repository wurde bereits geklont und enthält Trainingsdaten.

```bash
python train_dart_model.py
```

**Erwartete Dauer:**
- Mit GPU (NVIDIA): ~30-60 Minuten
- Mit CPU: 2-4 Stunden

Das trainierte Modell wird automatisch in `models/deepdarts_trained.pt` speichert und beide Neustart von BullSight geladen.

## Option 2: Ihr Eigenes Modell Trainieren

### Schritt 1: Daten Sammeln

1. BullSight öffnen und ""Calibration Screen"" navigieren
2. ""Capture Test Image"" klicken, um Dartboard-Fotos mit verschiedenen Dart-Positionen zu sammeln
3. Mindestens 50-100 Bilder sammeln für gute Resultate

### Schritt 2: Annotieren mit LabelImg

```bash
pip install labelimg
labelimg
```

1. ""Open Dir"" → Ordner mit Ihren Bildern wählen
2. Jedes Dart als ""dart"" mit Bounding Box annotieren
3. Optional: Calibration-Punkte mit ""cal_top"", ""cal_right"", ""cal_bottom"", ""cal_left"" annotieren

### Schritt 3: Dataset Strukturieren

```
my_dart_dataset/
├── images/
│   ├── train/  (70% Ihrer Bilder)
│   ├── val/    (20% Ihrer Bilder)
│   └── test/   (10% Ihrer Bilder)
├── labels/
│   ├── train/  (Entsprechende .txt Annotation-Dateien)
│   ├── val/
│   └── test/
└── data.yaml
```

### Schritt 4: data.yaml Erstellen

```yaml
path: /full/path/to/my_dart_dataset
train: images/train
val: images/val
test: images/test

nc: 1  # oder 5 wenn Sie auch Calibration-Punkte annotieren
names:
  0: dart
  # Optional für Calibration-Punkte:
  # 1: cal_top
  # 2: cal_right
  # 3: cal_bottom
  # 4: cal_left
```

### Schritt 5: Training-Script Erstellen

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # oder 's' für schnu, 'l' für höchste Genauigkeit

# Train
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,  # GPU number or 'cpu'
    patience=25,
    save=True,
)

# Copy best model to BullSight
import shutil
shutil.copy('runs/detect/train/weights/best.pt', 'path/to/BullSight/models/deepdarts_trained.pt')
```

### Schritt 6: In BullSight Verwenden

Das Modell wird automatisch beim nächsten Neustart geladen.

```bash
python -m src.main
```

## Modell-Größen Referenz

| Model | Parameter | Größe | Geschwindigkeit | Genauigkeit |
|-------|-----------|-------|-----------------|------------|
| YOLOv8n | 3.2M | 6.3 MB | ⚡ Sehr schnell | ~75% |
| YOLOv8s | 11.2M | 22.5 MB | ⚡⚡ Schnell | ~81% |
| YOLOv8m | 25.9M | 49.0 MB | ⚡⚡⚡ Mittel | ~83% |
| YOLOv8l | 43.7M | 83.3 MB | ⚡⚡⚡ Langsam | ~85% |

**Empfehlung:** 
- Für Raspberry Pi: `yolov8n` (Nano)
- Für Desktop: `yolov8s` (Small) - beste Balance

## Modell-Spezifikation für BullSight

Das BullSight-System erkennt automatisch das folgende Modell-Format:

```
models/
├── deepdarts_trained.pt         # Trainiertes Modell (bevorzugt)
├── deepdarts_s_best.pt          # Alternative Name
└── deepdarts_best.pt            # Alternative Name
```

Oder automatisches Training generiert:
```
bullsight_training/
└── dart_detector_v1/
    └── weights/
        └── best.pt
```

## Troubleshooting

### Modell wird nicht geladen
```python
# Überprüfe, ob Ultralytics verfügbar ist
python -c "from ultralytics import YOLO; print('OK')"
```

### Training ist zu langsam
- Verwende kleinere Batchgröße (`batch=8` statt 32)
- Verwende YOLOv8n statt `s`/`l`/`x`
- Verwende GPU (install `torch` mit CUDA)

### Schlechte Detektions-Ergebnisse
- Sammeln Sie mehr und vielfältigere Trainingsdaten
- Verwenden Sie mehr Epochs (>150)
- Verwenden Sie größeres Modell (YOLOv8s oder larger)
- Überprüfen Sie Ihrer Kamera-Kalibrierung

## Zusätzliche Ressourcen

- [Ultralytics Dokumentation](https://docs.ultralytics.com)
- [YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)
- [DartsVision Paper](https://arxiv.org/abs/2105.09880)
- [Original DartsVision Repository](https://github.com/mohamedamineyoukaoui-ops/DartsVision-)
