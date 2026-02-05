"""
Unit tests for BoardMapper calibration system.

Tests coordinate transformation, segment mapping, zone detection, and calibration.
Coverage Target: 100%

Author: Mario Neuhauser
"""

import pytest
import numpy as np
from pathlib import Path
import json

from src.calibration.board_mapper import (
    DartboardField,
    CalibrationData,
    BoardMapper,
    DARTBOARD_SEGMENTS,
    create_default_calibration
)


class TestDartboardField:
    """Tests for DartboardField dataclass."""
    
    def test_field_creation(self):
        """Test DartboardField initialization."""
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        assert field.segment == 20
        assert field.zone == "triple"
        assert field.score == 60
        assert field.multiplier == 3
    
    def test_bulls_eye_repr(self):
        """Test bull's eye string representation."""
        field = DartboardField(segment=25, zone="bull_eye", score=50, multiplier=1)
        assert "Bull's Eye (50)" in repr(field)
    
    def test_bull_repr(self):
        """Test bull string representation."""
        field = DartboardField(segment=25, zone="bull", score=25, multiplier=1)
        assert "Bull (25)" in repr(field)
    
    def test_miss_repr(self):
        """Test miss string representation."""
        field = DartboardField(segment=0, zone="miss", score=0, multiplier=0)
        assert "Miss (0)" in repr(field)
    
    def test_regular_field_repr(self):
        """Test regular field string representation."""
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        repr_str = repr(field)
        assert "Triple" in repr_str
        assert "20" in repr_str


class TestCalibrationData:
    """Tests for CalibrationData dataclass."""
    
    def test_calibration_creation(self):
        """Test CalibrationData initialization."""
        cal = CalibrationData(
            center_x=640,
            center_y=360,
            bull_eye_radius=10.0,
            bull_radius=25.0,
            triple_inner_radius=200.0,
            triple_outer_radius=220.0,
            double_inner_radius=350.0,
            double_outer_radius=380.0
        )
        assert cal.center_x == 640
        assert cal.center_y == 360
    
    def test_to_dict(self):
        """Test calibration serialization to dict."""
        cal = CalibrationData(
            center_x=640, center_y=360,
            bull_eye_radius=10.0, bull_radius=25.0,
            triple_inner_radius=200.0, triple_outer_radius=220.0,
            double_inner_radius=350.0, double_outer_radius=380.0
        )
        data = cal.to_dict()
        
        assert data["center_x"] == 640
        assert data["bull_eye_radius"] == 10.0
    
    def test_from_dict(self):
        """Test calibration deserialization from dict."""
        data = {
            "center_x": 640, "center_y": 360,
            "bull_eye_radius": 10.0, "bull_radius": 25.0,
            "triple_inner_radius": 200.0, "triple_outer_radius": 220.0,
            "double_inner_radius": 350.0, "double_outer_radius": 380.0
        }
        cal = CalibrationData.from_dict(data)
        
        assert cal.center_x == 640
        assert cal.bull_radius == 25.0


class TestBoardMapper:
    """Tests for BoardMapper class."""
    
    @pytest.fixture
    def calibration(self):
        """Create test calibration data."""
        return CalibrationData(
            center_x=640,
            center_y=360,
            bull_eye_radius=10.0,
            bull_radius=25.0,
            triple_inner_radius=200.0,
            triple_outer_radius=220.0,
            double_inner_radius=350.0,
            double_outer_radius=380.0
        )
    
    @pytest.fixture
    def mapper(self, calibration):
        """Create mapper with calibration."""
        return BoardMapper(calibration)
    
    def test_initialization_without_calibration(self):
        """Test mapper initialization without calibration."""
        mapper = BoardMapper()
        assert mapper.calibration is None
        assert mapper.segments == DARTBOARD_SEGMENTS
    
    def test_initialization_with_calibration(self, calibration):
        """Test mapper initialization with calibration."""
        mapper = BoardMapper(calibration)
        assert mapper.calibration == calibration
    
    def test_set_calibration(self, calibration):
        """Test setting calibration after initialization."""
        mapper = BoardMapper()
        mapper.set_calibration(calibration)
        assert mapper.calibration == calibration
    
    def test_load_calibration(self, calibration, tmp_path):
        """Test loading calibration from file."""
        # Save calibration
        filepath = tmp_path / "calibration.json"
        with open(filepath, 'w') as f:
            json.dump(calibration.to_dict(), f)
        
        # Load it
        mapper = BoardMapper()
        mapper.load_calibration(filepath)
        
        assert mapper.calibration.center_x == calibration.center_x
    
    def test_load_calibration_file_not_found(self, tmp_path):
        """Test loading non-existent calibration file."""
        mapper = BoardMapper()
        filepath = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            mapper.load_calibration(filepath)
    
    def test_load_calibration_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        mapper = BoardMapper()
        filepath = tmp_path / "invalid.json"
        filepath.write_text("not valid json{")
        
        with pytest.raises(ValueError, match="Invalid calibration JSON"):
            mapper.load_calibration(filepath)
    
    def test_save_calibration(self, mapper, tmp_path):
        """Test saving calibration to file."""
        filepath = tmp_path / "calibration.json"
        mapper.save_calibration(filepath)
        
        assert filepath.exists()
        
        # Verify content
        with open(filepath) as f:
            data = json.load(f)
        assert data["center_x"] == 640
    
    def test_save_calibration_without_data(self, tmp_path):
        """Test saving when no calibration is set."""
        mapper = BoardMapper()
        filepath = tmp_path / "calibration.json"
        
        with pytest.raises(RuntimeError, match="No calibration data"):
            mapper.save_calibration(filepath)
    
    def test_pixel_to_polar_center(self, mapper):
        """Test polar conversion at center."""
        radius, angle = mapper.pixel_to_polar(640, 360)
        
        assert radius == pytest.approx(0, abs=1)
    
    def test_pixel_to_polar_right(self, mapper):
        """Test polar conversion to the right (segment 6 direction)."""
        radius, angle = mapper.pixel_to_polar(740, 360)
        
        assert radius == pytest.approx(100, abs=1)
        assert angle == pytest.approx(0, abs=5)  # Right is 0°
    
    def test_pixel_to_polar_top(self, mapper):
        """Test polar conversion upward (segment 20 direction)."""
        radius, angle = mapper.pixel_to_polar(640, 260)
        
        assert radius == pytest.approx(100, abs=1)
        assert angle == pytest.approx(270, abs=5)  # Up is 270°
    
    def test_pixel_to_polar_without_calibration(self):
        """Test error when converting without calibration."""
        mapper = BoardMapper()
        
        with pytest.raises(RuntimeError, match="Calibration must be set"):
            mapper.pixel_to_polar(100, 100)
    
    def test_angle_to_segment_20(self, mapper):
        """Test angle mapping to segment 20 (top)."""
        segment = mapper.angle_to_segment(270)  # Up/top
        # Segment 20 should be at ~270°
        assert segment == 20
    
    def test_angle_to_segment_6(self, mapper):
        """Test angle mapping to segment 6 (right)."""
        segment = mapper.angle_to_segment(0)  # Right
        # Segment 6 starts at ~-9° (or 351°)
        assert segment == 6
    
    def test_angle_to_segment_3(self, mapper):
        """Test angle mapping to segment 3 (bottom)."""
        segment = mapper.angle_to_segment(90)  # Down/bottom
        assert segment == 3
    
    def test_radius_to_zone_bulls_eye(self, mapper):
        """Test zone detection for bull's eye."""
        zone = mapper.radius_to_zone(5)
        assert zone == "bull_eye"
    
    def test_radius_to_zone_bull(self, mapper):
        """Test zone detection for bull."""
        zone = mapper.radius_to_zone(20)
        assert zone == "bull"
    
    def test_radius_to_zone_single(self, mapper):
        """Test zone detection for inner single."""
        zone = mapper.radius_to_zone(150)
        assert zone == "single"
    
    def test_radius_to_zone_outer_single(self, mapper):
        """Test zone detection for outer single (between triple and double)."""
        zone = mapper.radius_to_zone(300)  # Between triple_outer (220) and double_inner (350)
        assert zone == "single"
    
    def test_radius_to_zone_triple(self, mapper):
        """Test zone detection for triple."""
        zone = mapper.radius_to_zone(210)
        assert zone == "triple"
    
    def test_radius_to_zone_double(self, mapper):
        """Test zone detection for double."""
        zone = mapper.radius_to_zone(370)
        assert zone == "double"
    
    def test_radius_to_zone_miss(self, mapper):
        """Test zone detection for miss."""
        zone = mapper.radius_to_zone(400)
        assert zone == "miss"
    
    def test_radius_to_zone_without_calibration(self):
        """Test error when detecting zone without calibration."""
        mapper = BoardMapper()
        
        with pytest.raises(RuntimeError, match="Calibration must be set"):
            mapper.radius_to_zone(100)
    
    def test_calculate_score_bulls_eye(self, mapper):
        """Test score calculation for bull's eye."""
        score, multiplier = mapper.calculate_score(25, "bull_eye")
        assert score == 50
        assert multiplier == 1
    
    def test_calculate_score_bull(self, mapper):
        """Test score calculation for bull."""
        score, multiplier = mapper.calculate_score(25, "bull")
        assert score == 25
        assert multiplier == 1
    
    def test_calculate_score_miss(self, mapper):
        """Test score calculation for miss."""
        score, multiplier = mapper.calculate_score(0, "miss")
        assert score == 0
        assert multiplier == 0
    
    def test_calculate_score_single(self, mapper):
        """Test score calculation for single."""
        score, multiplier = mapper.calculate_score(20, "single")
        assert score == 20
        assert multiplier == 1
    
    def test_calculate_score_double(self, mapper):
        """Test score calculation for double."""
        score, multiplier = mapper.calculate_score(20, "double")
        assert score == 40
        assert multiplier == 2
    
    def test_calculate_score_triple(self, mapper):
        """Test score calculation for triple."""
        score, multiplier = mapper.calculate_score(20, "triple")
        assert score == 60
        assert multiplier == 3
    
    def test_map_coordinate_bulls_eye(self, mapper):
        """Test complete mapping to bull's eye."""
        field = mapper.map_coordinate(640, 360)  # Center
        
        assert field.segment == 25
        assert field.zone == "bull_eye"
        assert field.score == 50
    
    def test_map_coordinate_triple_20(self, mapper):
        """Test complete mapping to triple 20."""
        field = mapper.map_coordinate(640, 150)  # Top, in triple ring
        
        assert field.segment == 20
        assert field.zone == "triple"
        assert field.score == 60
    
    def test_map_coordinate_miss(self, mapper):
        """Test complete mapping to miss."""
        field = mapper.map_coordinate(100, 100)  # Far from board
        
        assert field.zone == "miss"
        assert field.score == 0
    
    def test_map_coordinate_without_calibration(self):
        """Test error when mapping without calibration."""
        mapper = BoardMapper()
        
        with pytest.raises(RuntimeError):
            mapper.map_coordinate(100, 100)
    
    def test_validate_calibration_valid(self, mapper):
        """Test validation of valid calibration."""
        checks = mapper.validate_calibration()
        
        assert checks["calibration_set"] is True
        assert checks["all_checks_passed"] is True
    
    def test_validate_calibration_invalid_radii_order(self, calibration):
        """Test validation detects invalid radii order."""
        calibration.triple_outer_radius = 100  # Smaller than inner
        mapper = BoardMapper(calibration)
        
        checks = mapper.validate_calibration()
        
        assert checks["triple_inner_smaller_than_outer"] is False
        assert checks["all_checks_passed"] is False
    
    def test_validate_calibration_not_set(self):
        """Test validation when no calibration is set."""
        mapper = BoardMapper()
        checks = mapper.validate_calibration()
        
        assert checks["calibration_set"] is False
    
    def test_get_segment_bounds(self, mapper):
        """Test getting angle boundaries for segment."""
        start, end = mapper.get_segment_bounds(20)
        
        # Segment 20 is at top, should span around 270°
        assert isinstance(start, float)
        assert isinstance(end, float)
    
    def test_get_segment_bounds_invalid(self, mapper):
        """Test error for invalid segment."""
        with pytest.raises(ValueError, match="Invalid segment"):
            mapper.get_segment_bounds(99)


class TestCreateDefaultCalibration:
    """Tests for create_default_calibration helper."""
    
    def test_default_calibration_creation(self):
        """Test creating default calibration."""
        cal = create_default_calibration(1280, 720, 300)
        
        assert cal.center_x == 640
        assert cal.center_y == 360
        assert cal.double_outer_radius == pytest.approx(300, abs=1)
    
    def test_default_calibration_proportions(self):
        """Test default calibration maintains proper proportions."""
        cal = create_default_calibration(1280, 720, 300)
        
        # Check radius ordering
        assert cal.bull_eye_radius < cal.bull_radius
        assert cal.bull_radius < cal.triple_inner_radius
        assert cal.triple_inner_radius < cal.triple_outer_radius
        assert cal.triple_outer_radius < cal.double_inner_radius
        assert cal.double_inner_radius < cal.double_outer_radius


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.calibration", "--cov-report=term-missing"])
