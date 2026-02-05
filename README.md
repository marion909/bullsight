# 🎯 BullSight - Automated Dart Scoring System

Raspberry Pi-based dart scoring system with computer vision and touch interface.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest`
3. See [Phase 1](phase1-foundations.md) for setup details

## Phases
- [Phase 1: Foundations](phase1-foundations.md) ✅ In Progress
- [Phase 2: Vision Engine](phase2-vision.md)
- [Phase 3: Calibration](phase3-calibration.md)
- [Phase 4: Game Engine](phase4-game-engine.md)
- [Phase 5: UI & Polish](phase5-ui-polish.md)

## Project Structure
```
BullSight/
├── src/
│   ├── vision/          # Computer vision components
│   ├── calibration/     # Dartboard mapping
│   ├── game/            # Game engine logic
│   ├── ui/              # PyQt6 interface
│   └── config/          # Configuration management
├── tests/
│   ├── unit/            # Unit tests (isolated)
│   ├── integration/     # Integration tests
│   └── e2e/             # End-to-end tests
├── docs/                # API documentation
├── config/              # JSON/YAML configurations
├── assets/              # Images and sounds
└── logs/                # Application logs
```

## Development Setup

### Prerequisites
- Python 3.11+
- Raspberry Pi 4 or 5 (4 GB RAM)
- Raspberry Pi Camera Module v3
- 7-inch Touch Display

### Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
```

## Testing
```bash
# Run all tests with coverage
pytest

# Run specific test category
pytest -m unit
pytest -m integration
pytest -m e2e

# Generate HTML coverage report
pytest --cov --cov-report=html
```

## Current Status
- **Phase:** 1 - Foundations
- **Status:** 🟡 In Progress
- **Coverage:** TBD

**Author:** Mario Neuhauser  
**Platform:** Raspberry Pi 4/5  
**Language:** Python 3.11+
