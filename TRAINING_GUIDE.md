# 🎯 BullSight Erkennungs-Training & Optimierung

## Überblick

BullSight verwendet **differenzbasierte Computer Vision** (nicht Machine Learning). Die "Training" besteht aus:
1. Erstellen eines perfekten Referenzbildes
2. Optimieren der Erkennungsparameter
3. Testen unter verschiedenen Bedingungen

## 📸 Schritt 1: Perfektes Referenzbild erstellen

### Voraussetzungen
- ✅ Dartboard ist vollständig leer (keine Pfeile!)
- ✅ Beleuchtung ist optimal und gleichmäßig
- ✅ Kamera ist montiert und fokussiert
- ✅ Keine Schatten auf dem Board

### Methode A: Über die UI (empfohlen)

1. Starte BullSight: `./run.sh` (Linux/Raspberry Pi) oder `.\run.bat` (Windows)
2. Navigiere zu **Settings** → **Calibration**
3. Stelle sicher, dass das Dartboard **komplett leer** ist (keine Darts!)
4. Klicke auf **"Capture Reference Image"**
5. Bestätige den Dialog
6. Warte 3 Sekunden während die Kamera fokussiert
7. ✅ Referenzbild wird automatisch gespeichert nach `config/reference_board.jpg`

Die UI-Methode:
- Triggert automatisch Autofokus
- Nimmt 10 Frames und wählt das stabilste
- Speichert direkt im korrekten Format
- Zeigt Erfolgsmeldung mit Speicherort

### Methode B: Manuell via Python

```python
from src.vision.camera_manager import CameraManager
from src.vision.dart_detector import DartDetector
from pathlib import Path

# 1. Initialisiere Komponenten
camera = CameraManager(resolution=(1280, 720), enable_autofocus=True)
detector = DartDetector()

# 2. Starte Kamera und fokussiere
camera.start()
camera.trigger_autofocus()  # Warte 2-3 Sekunden
import time
time.sleep(3)

# 3. Capture mehrere Frames und nimm das beste
frames = []
for i in range(10):
    frame = camera.capture_frame()
    frames.append(frame)
    time.sleep(0.1)

# 4. Wähle Frame mit geringstem Rauschen (mittlere Frame)
reference_frame = frames[5]

# 5. Setze als Referenz und speichere
detector.set_reference_image(reference_frame)
detector.save_reference_to_file("config/reference_board.jpg")

# 6. Aufräumen
camera.stop()

print("✅ Referenzbild gespeichert!")
```

### Methode C: Über Terminal

```bash
# Erstelle ein Script
cat > capture_reference.py << 'EOF'
import sys
sys.path.insert(0, '.')
from src.vision.camera_manager import CameraManager
from src.vision.dart_detector import DartDetector
import time

camera = CameraManager()
detector = DartDetector()

print("📸 Capturing reference image in 5 seconds...")
print("   Make sure dartboard is EMPTY!")
time.sleep(5)

camera.start()
time.sleep(2)  # Autofocus

frame = camera.capture_frame()
detector.set_reference_image(frame)
detector.save_reference_to_file("config/reference_board.jpg")

camera.stop()
print("✅ Reference image saved to config/reference_board.jpg")
EOF

# Ausführen
export PYTHONPATH="$(pwd)"
python capture_reference.py
```

## ⚙️ Schritt 2: Parameter optimieren

### Die wichtigsten Parameter

```python
detector = DartDetector(
    min_contour_area=100,      # Minimum Dart-Größe
    max_contour_area=5000,     # Maximum Dart-Größe
    blur_kernel_size=5,        # Rauschfilter
    threshold_value=30         # Empfindlichkeit
)
```

### Parameter-Guide

#### 1. `threshold_value` (Empfindlichkeit)
**Was es tut**: Bestimmt, wie groß der Unterschied sein muss

- **Zu niedrig (10-20)**: Erkennt zu viele falsche Darts (Schatten, Bewegungen)
- **Optimal (25-40)**: Zuverlässige Erkennung
- **Zu hoch (50+)**: Verpasst echte Darts

**Anpassen für:**
- 🔆 Helle Umgebung: 35-45
- 🌙 Dunkle Umgebung: 25-35
- 💡 Wechselndes Licht: 30-40

#### 2. `min_contour_area` (Minimalgröße)
**Was es tut**: Filtert kleine Störungen heraus

- **Zu niedrig (<50)**: Viele Fehlerkennungen
- **Optimal (100-200)**: Gut für Standard-Darts
- **Zu hoch (>300)**: Verpasst Dart-Spitzen

**Anpassen für:**
- Entfernung Kamera → Board: Näher = größere Werte
- Dart-Typ: Dünne Spitzen = kleinere Werte

#### 3. `max_contour_area` (Maximalgröße)
**Was es tut**: Filtert große Objekte (Hand, Schatten)

- **Zu niedrig (<3000)**: Verpasst breite Darts
- **Optimal (4000-6000)**: Standard-Darts
- **Zu hoch (>8000)**: Erkennt Hände/Schatten

#### 4. `blur_kernel_size` (Rauschfilter)
**Was es tut**: Glättet Bild vor Vergleich

- **Klein (3)**: Mehr Details, mehr Rauschen
- **Optimal (5-7)**: Gute Balance
- **Groß (9+)**: Weniger Rauschen, weniger Details

## 🧪 Schritt 3: Testing & Optimierung

### Test-Script erstellen

```python
# test_detection.py
import sys
sys.path.insert(0, '.')
from src.vision.camera_manager import CameraManager
from src.vision.dart_detector import DartDetector
import time

# Parameter zum Testen
TEST_PARAMS = [
    {"threshold": 25, "min_area": 100, "max_area": 5000},
    {"threshold": 30, "min_area": 150, "max_area": 5000},
    {"threshold": 35, "min_area": 100, "max_area": 4000},
]

camera = CameraManager()
camera.start()

# Lade Referenzbild
reference_detector = DartDetector()
reference_detector.load_reference_from_file("config/reference_board.jpg")

print("🎯 Wirf jetzt einen Dart!")
time.sleep(5)

current_frame = camera.capture_frame()

# Teste verschiedene Parameter
for i, params in enumerate(TEST_PARAMS, 1):
    detector = DartDetector(
        min_contour_area=params["min_area"],
        max_contour_area=params["max_area"],
        threshold_value=params["threshold"]
    )
    detector.set_reference_image(reference_detector.reference_image)
    
    result = detector.detect_dart(current_frame)
    
    print(f"\n--- Test {i} ---")
    print(f"Parameters: {params}")
    if result:
        print(f"✅ Dart detected at ({result.x}, {result.y})")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Area: {result.contour_area:.0f}")
    else:
        print("❌ No dart detected")

camera.stop()
```

### Systematisches Testen

```bash
# 1. Verschiedene Beleuchtungen
./test_detection.py  # Tageslicht
./test_detection.py  # Kunstlicht
./test_detection.py  # Dämmerlicht

# 2. Verschiedene Positionen
# Dart in Bullseye
# Dart in Triple 20
# Dart in Double-Ring
# Dart am Rand

# 3. Verschiedene Dart-Typen
# Steeldarts (dünn)
# Softdarts (dick)
# Verschiedene Farben
```

## 📊 Schritt 4: Live-Tuning

### Visualisierung während der Erkennung

```python
# live_tuning.py
import sys
sys.path.insert(0, '.')
from src.vision.camera_manager import CameraManager
from src.vision.dart_detector import DartDetector
import cv2

camera = CameraManager()
detector = DartDetector()
detector.load_reference_from_file("config/reference_board.jpg")

camera.start()

print("🎯 Live Tuning Mode")
print("   Adjust parameters and see results in real-time")
print("   Press 'q' to quit")

# Trackbars für Parameter
cv2.namedWindow("Tuning")
cv2.createTrackbar("Threshold", "Tuning", 30, 100, lambda x: None)
cv2.createTrackbar("Min Area", "Tuning", 100, 500, lambda x: None)
cv2.createTrackbar("Max Area", "Tuning", 5000, 10000, lambda x: None)

while True:
    # Hole aktuelle Parameter
    threshold = cv2.getTrackbarPos("Threshold", "Tuning")
    min_area = cv2.getTrackbarPos("Min Area", "Tuning")
    max_area = cv2.getTrackbarPos("Max Area", "Tuning")
    
    # Update Detector
    detector.threshold_value = threshold
    detector.min_contour_area = min_area
    detector.max_contour_area = max_area
    
    # Capture und erkenne
    frame = camera.capture_frame()
    result = detector.detect_dart(frame)
    
    # Visualisiere
    vis = detector.visualize_detection(frame, result)
    
    # Zeige Info
    info_text = f"T:{threshold} Min:{min_area} Max:{max_area}"
    cv2.putText(vis, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if result:
        cv2.putText(vis, f"Dart at ({result.x}, {result.y})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Tuning", vis)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()

print(f"\n✅ Optimale Parameter gefunden:")
print(f"   threshold_value={threshold}")
print(f"   min_contour_area={min_area}")
print(f"   max_contour_area={max_area}")
```

## 💡 Best Practices

### Beleuchtung
- ✅ **Gleichmäßige Ausleuchtung**: Keine Schatten
- ✅ **Konstante Beleuchtung**: Keine wechselnden Lichtverhältnisse
- ✅ **Kein Blitz**: Kontinuierliches Licht
- ❌ **Vermeiden**: Direkte Sonne, blinkende Lichter

### Kamera-Setup
- ✅ **Autofokus aktiviert**: Scharfes Bild
- ✅ **Feste Montierung**: Keine Bewegung
- ✅ **Optimaler Abstand**: 50-100cm vom Board
- ✅ **Zentrierte Ansicht**: Board in Bildmitte

### Referenzbild
- ✅ **Komplett leer**: Keine Darts, keine Hände
- ✅ **Alle Segmente sichtbar**: Komplettes Board
- ✅ **Scharf**: Kein Motion Blur
- ✅ **Regelmäßig erneuern**: Bei Lichtwechsel neu erstellen

### Erkennungs-Qualität
- **Gut**: 95%+ korrekte Erkennungen
- **Akzeptabel**: 85-95% korrekt
- **Schlecht**: <85% → Parameter anpassen

## 🔧 Erweiterte Optimierung

### Adaptive Threshold
Für wechselnde Lichtverhältnisse:

```python
# Implementiere adaptive threshold
def auto_threshold(frame, reference):
    """Berechne optimalen Threshold basierend auf Bildhelligkeit"""
    avg_brightness = np.mean(frame)
    ref_brightness = np.mean(reference)
    
    brightness_diff = abs(avg_brightness - ref_brightness)
    
    if brightness_diff < 10:
        return 30  # Normal
    elif brightness_diff < 30:
        return 35  # Leichter Unterschied
    else:
        return 40  # Großer Unterschied
```

### Multi-Frame-Validierung
Reduziere Fehlerkennungen:

```python
def detect_with_validation(detector, camera, num_frames=3):
    """Erkenne Dart nur wenn in mehreren Frames erkannt"""
    detections = []
    
    for _ in range(num_frames):
        frame = camera.capture_frame()
        result = detector.detect_dart(frame)
        if result:
            detections.append(result)
        time.sleep(0.1)
    
    if len(detections) >= num_frames - 1:  # 2 von 3
        # Mittelwert der Positionen
        avg_x = sum(d.x for d in detections) / len(detections)
        avg_y = sum(d.y for d in detections) / len(detections)
        return DartCoordinate(int(avg_x), int(avg_y), 1.0, 0)
    
    return None
```

## 📈 Monitoring & Logging

### Erkennungs-Statistiken sammeln

```python
class DetectionStats:
    def __init__(self):
        self.total_attempts = 0
        self.successful_detections = 0
        self.false_positives = 0
        
    def log_detection(self, detected: bool, validated: bool):
        self.total_attempts += 1
        if detected:
            self.successful_detections += 1
            if not validated:
                self.false_positives += 1
    
    def accuracy(self):
        if self.total_attempts == 0:
            return 0
        return (self.successful_detections - self.false_positives) / self.total_attempts

# Verwendung
stats = DetectionStats()
# ... bei jeder Erkennung:
stats.log_detection(detected=True, validated=True)
print(f"Accuracy: {stats.accuracy():.1%}")
```

## 🎓 Troubleshooting

### Problem: Zu viele Fehlerkennungen
**Lösung:**
- Erhöhe `threshold_value` (30 → 40)
- Erhöhe `min_contour_area` (100 → 200)
- Verbessere Beleuchtung
- Erstelle neues Referenzbild

### Problem: Darts werden nicht erkannt
**Lösung:**
- Senke `threshold_value` (30 → 25)
- Senke `min_contour_area` (100 → 80)
- Prüfe Fokus der Kamera
- Prüfe ob Referenzbild aktuell ist

### Problem: Schatten werden als Darts erkannt
**Lösung:**
- Optimiere Beleuchtung (keine Schatten)
- Erhöhe `threshold_value`
- Verwende Multi-Frame-Validierung

### Problem: Inkonsistente Erkennungen
**Lösung:**
- Fixiere Kamera besser (keine Bewegung)
- Verwende höheren `blur_kernel_size`
- Erstelle neues Referenzbild bei gleichem Licht

## 📚 Weiterführende Optimierungen

### Zukünftige Features (optional)
- **Maschinelles Lernen**: YOLOv8 für Dart-Erkennung
- **Mehrere Referenzbilder**: Für verschiedene Lichtsituationen
- **Automatische Parameter-Anpassung**: Selbst-Kalibrierung
- **Dart-Tracking**: Bewegungsverfolgung statt Differenz

---

**Dokumentation erstellt für BullSight v1.0**
**Autor: Mario Neuhauser**
