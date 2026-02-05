# 🎯 Dart Scoring System – Raspberry Pi

Automatisches Dart-Punktesystem mit Kamera, Lichtring und Touch-Display auf einem Raspberry Pi.

---

## 📌 Projektziel

Ein standalone Dart-Scoring-System, das:
- Dartwürfe per **Kamera** erkennt
- **Punkte automatisch berechnet**
- Über ein **Touch-Display** bedient wird
- Alle gängigen **Dart-Regeln & Spielmodi** unterstützt
- Lokal, ohne Cloud-Zwang, läuft

---

## 🧱 Hardware-Anforderungen

### Pflicht
- Raspberry Pi 4 oder 5 (4 GB RAM empfohlen)
- Raspberry Pi Camera Module v3 (Autofokus)
- Touch-Display (7 Zoll empfohlen)
- LED-Lichtring (gleichmäßige Ausleuchtung)
- Standard-Steel-Dartboard

### Optional
- Lautsprecher (Soundeffekte)
- Gehäuse / 3D-gedruckte Halterung
- Externer Power-Button

---

## 📷 Kameraposition & Setup

- Kamera **zentral vor der Dartscheibe**
- Integriert im oder hinter dem Lichtring
- Abstand: ca. 25–40 cm
- Kamera exakt **senkrecht zur Scheibe**
- Dartboard muss fest montiert sein

Warum zentral?
- Minimale Verzerrung
- Vereinfachte Geometrie
- Bessere Treffererkennung

---

## 🧠 Software-Architektur (Übersicht)

- **UI Layer** (Touch)
- **Game Engine** (Dart-Regeln)
- **Vision Engine** (OpenCV)
- **Mapping & Kalibrierung**
- **Config & Persistence**

Datenfluss:

Kamera → Treffererkennung → Koordinate → Dartfeld → Punkte → Game Engine → UI

---

## 👁️ Computer Vision

### Tech-Stack
- Python 3
- OpenCV
- NumPy
- Optional: TensorFlow Lite (nur falls nötig)

### Erkennungsstrategie (ohne Machine Learning)

1. Referenzbild **ohne Dart** speichern
2. Nach jedem Wurf neues Bild aufnehmen
3. Differenzbild erzeugen
4. Konturen erkennen
5. Dartspitze bestimmen
6. Pixel-Koordinate extrahieren

Vorteile:
- Schnell
- Stabil
- Auf Raspberry Pi gut lauffähig

---

## 🗺️ Kalibrierung & Mapping

### Kalibrierung (über Touch-UI)

- Mittelpunkt der Scheibe festlegen
- Bull-Radius definieren
- Double- & Triple-Ringe bestimmen
- Segmentwinkel automatisch berechnen

### Mathematisches Mapping

- Winkel → Segment (20, 1, 18, ...)
- Radius → Single / Double / Triple / Bull

---

## 🎮 Game Engine

### Unterstützte Spielmodi (geplant)

- 301 / 501 / 701
- Double-In / Double-Out
- Master-Out
- Cricket
- Around the Clock
- Trainingsmodus

### Features
- Mehrspielerfähig
- Runden- & Wurfverwaltung
- Bust-Logik
- Statistiken (Avg, Checkout-Quote)

---

## 📱 Touch-UI

### Framework
- PyQt5 / PyQt6 (empfohlen)

### Screens
- Start / Spielauswahl
- Spielerverwaltung
- Live-Score-Anzeige
- Kalibrierung
- Einstellungen

### UX-Ziele
- Große Buttons
- Dart-tauglich (kein Präzisionstippen)
- Schnelle Reaktion

---

## ⚙️ Konfiguration

### Format
- JSON oder YAML

### Inhalte
- Spielregeln
- Spieleranzahl
- Kameraeinstellungen
- Board-Typ
- UI-Optionen

---

## 🛠️ Entwicklungs-Roadmap

### Phase 1 – Grundlagen
- Raspberry Pi OS Setup
- Kamera-Test
- Touch-Display Integration
- Projektstruktur anlegen

### Phase 2 – Vision Prototyp
- Live-Kamera-Feed
- Referenzbild speichern
- Dart-Erkennung
- Koordinaten bestimmen

### Phase 3 – Mapping & Kalibrierung
- Kalibrierungs-UI
- Segment-Berechnung
- Punktelogik testen

### Phase 4 – Game Engine
- 501-Spiel komplett spielbar
- Mehrspieler
- Regeloptionen

### Phase 5 – UI & Feinschliff
- Saubere UI
- Soundeffekte
- Statistiken
- Fehlerbehandlung

---

## ⚠️ Risiken & Lösungen

| Risiko | Lösung |
|------|------|
| Schlechte Erkennung | Starker, diffuser Lichtring |
| Schatten | Gleichmäßige Ausleuchtung |
| Dart bleibt stecken | Referenzbild nach jedem Wurf |
| Pi zu langsam | Auflösung reduzieren |
| Verzerrung | Exakte Kameraposition |

---

## 📈 Erweiterungen (optional)

- Online-Multiplayer
- Spielerprofile
- Export von Statistiken
- App-Anbindung
- KI-Wurfanalyse

---

## ⏱️ Aufwandsschätzung

- Prototyp: 2–4 Wochen
- Stabiler MVP: 1–2 Monate
- Produktreif: 2–3 Monate

---

## ✅ Status

- [ ] Hardware final
- [ ] Vision Prototyp
- [ ] Kalibrierung
- [ ] Game Engine
- [ ] UI Final
- [ ] Release

---

**Autor:** Mario Neuhauser  
**Plattform:** Raspberry Pi  
**Sprache:** Python

