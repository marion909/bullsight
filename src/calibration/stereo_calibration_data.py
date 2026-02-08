"""
Stereo Calibration Data Structure.

Stores intrinsic and extrinsic parameters for stereo camera system,
including camera matrices, distortion coefficients, rotation, and translation.

Author: Mario Neuhauser
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class StereoCalibrationData:
    """
    Stores complete stereo calibration parameters.
    
    Intrinsic Parameters (per camera):
        K1, K2: 3x3 camera matrices (focal length, principal point)
        D1, D2: Distortion coefficients (k1, k2, p1, p2, k3)
    
    Extrinsic Parameters (camera relationship):
        R: 3x3 rotation matrix (right camera relative to left)
        T: 3x1 translation vector (right camera relative to left)
        E: 3x3 essential matrix
        F: 3x3 fundamental matrix
    
    Rectification Parameters (for stereo matching):
        R1, R2: 3x3 rectification transforms
        P1, P2: 3x4 projection matrices
        Q: 4x4 disparity-to-depth mapping matrix
    
    Metadata:
        image_size: (width, height) used during calibration
        rms_error: Root mean square reprojection error
        calibration_date: ISO timestamp
    """
    
    # Intrinsic parameters
    K1: np.ndarray  # Left camera matrix (3x3)
    K2: np.ndarray  # Right camera matrix (3x3)
    D1: np.ndarray  # Left distortion coefficients (5,)
    D2: np.ndarray  # Right distortion coefficients (5,)
    
    # Extrinsic parameters
    R: np.ndarray   # Rotation matrix (3x3)
    T: np.ndarray   # Translation vector (3,)
    E: np.ndarray   # Essential matrix (3x3)
    F: np.ndarray   # Fundamental matrix (3x3)
    
    # Rectification parameters
    R1: np.ndarray  # Left rectification transform (3x3)
    R2: np.ndarray  # Right rectification transform (3x3)
    P1: np.ndarray  # Left projection matrix (3x4)
    P2: np.ndarray  # Right projection matrix (3x4)
    Q: np.ndarray   # Disparity-to-depth matrix (4x4)
    
    # Metadata
    image_size: tuple  # (width, height)
    rms_error: float   # Reprojection error
    calibration_date: str  # ISO timestamp
    
    def __post_init__(self):
        """Validate matrix shapes after initialization."""
        self._validate_shapes()
    
    def _validate_shapes(self):
        """Validate all matrices have correct shapes."""
        assert self.K1.shape == (3, 3), f"K1 must be 3x3, got {self.K1.shape}"
        assert self.K2.shape == (3, 3), f"K2 must be 3x3, got {self.K2.shape}"
        assert self.D1.shape == (5,), f"D1 must be (5,), got {self.D1.shape}"
        assert self.D2.shape == (5,), f"D2 must be (5,), got {self.D2.shape}"
        assert self.R.shape == (3, 3), f"R must be 3x3, got {self.R.shape}"
        assert self.T.shape == (3,), f"T must be (3,), got {self.T.shape}"
        assert self.E.shape == (3, 3), f"E must be 3x3, got {self.E.shape}"
        assert self.F.shape == (3, 3), f"F must be 3x3, got {self.F.shape}"
        assert self.R1.shape == (3, 3), f"R1 must be 3x3, got {self.R1.shape}"
        assert self.R2.shape == (3, 3), f"R2 must be 3x3, got {self.R2.shape}"
        assert self.P1.shape == (3, 4), f"P1 must be 3x4, got {self.P1.shape}"
        assert self.P2.shape == (3, 4), f"P2 must be 3x4, got {self.P2.shape}"
        assert self.Q.shape == (4, 4), f"Q must be 4x4, got {self.Q.shape}"
        assert len(self.image_size) == 2, f"image_size must be (width, height)"
    
    def get_baseline(self) -> float:
        """
        Calculate stereo baseline distance in same units as translation vector.
        
        Returns:
            Baseline distance between camera centers
        """
        return np.linalg.norm(self.T)
    
    def get_focal_length_avg(self) -> float:
        """
        Get average focal length from both cameras.
        
        Returns:
            Average focal length in pixels
        """
        fx_left = self.K1[0, 0]
        fy_left = self.K1[1, 1]
        fx_right = self.K2[0, 0]
        fy_right = self.K2[1, 1]
        return (fx_left + fy_left + fx_right + fy_right) / 4
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary with all parameters (numpy arrays converted to lists)
        """
        return {
            "K1": self.K1.tolist(),
            "K2": self.K2.tolist(),
            "D1": self.D1.tolist(),
            "D2": self.D2.tolist(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist(),
            "R1": self.R1.tolist(),
            "R2": self.R2.tolist(),
            "P1": self.P1.tolist(),
            "P2": self.P2.tolist(),
            "Q": self.Q.tolist(),
            "image_size": self.image_size,
            "rms_error": self.rms_error,
            "calibration_date": self.calibration_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StereoCalibrationData':
        """
        Create instance from dictionary.
        
        Args:
            data: Dictionary with calibration parameters
            
        Returns:
            StereoCalibrationData instance
        """
        return cls(
            K1=np.array(data["K1"]),
            K2=np.array(data["K2"]),
            D1=np.array(data["D1"]),
            D2=np.array(data["D2"]),
            R=np.array(data["R"]),
            T=np.array(data["T"]),
            E=np.array(data["E"]),
            F=np.array(data["F"]),
            R1=np.array(data["R1"]),
            R2=np.array(data["R2"]),
            P1=np.array(data["P1"]),
            P2=np.array(data["P2"]),
            Q=np.array(data["Q"]),
            image_size=tuple(data["image_size"]),
            rms_error=data["rms_error"],
            calibration_date=data["calibration_date"]
        )
    
    def save(self, filepath: Path) -> None:
        """
        Save calibration to JSON file.
        
        Args:
            filepath: Path to save file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Stereo calibration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> Optional['StereoCalibrationData']:
        """
        Load calibration from JSON file.
        
        Args:
            filepath: Path to calibration file
            
        Returns:
            StereoCalibrationData instance or None if file doesn't exist
        """
        if not filepath.exists():
            logger.warning(f"Stereo calibration file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            calib = cls.from_dict(data)
            logger.info(f"Stereo calibration loaded from {filepath}")
            return calib
            
        except Exception as e:
            logger.error(f"Failed to load stereo calibration: {e}")
            return None
    
    def __str__(self) -> str:
        """String representation with key parameters."""
        baseline = self.get_baseline()
        focal = self.get_focal_length_avg()
        return (f"StereoCalibration("
                f"baseline={baseline:.1f}mm, "
                f"focal={focal:.1f}px, "
                f"rms_error={self.rms_error:.3f}, "
                f"size={self.image_size})")


def create_stereo_calibration_data(
    camera_matrix_left: np.ndarray,
    dist_coeffs_left: np.ndarray,
    camera_matrix_right: np.ndarray,
    dist_coeffs_right: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    F: np.ndarray,
    image_size: tuple,
    rms_error: float
) -> StereoCalibrationData:
    """
    Create StereoCalibrationData with rectification computed automatically.
    
    Args:
        camera_matrix_left: Left camera matrix (3x3)
        dist_coeffs_left: Left distortion coefficients
        camera_matrix_right: Right camera matrix (3x3)
        dist_coeffs_right: Right distortion coefficients
        R: Rotation matrix (3x3)
        T: Translation vector (3,)
        E: Essential matrix (3x3)
        F: Fundamental matrix (3x3)
        image_size: (width, height)
        rms_error: Reprojection error
        
    Returns:
        StereoCalibrationData with computed rectification parameters
    """
    import cv2
    from datetime import datetime
    
    # Compute rectification transforms
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0  # 0 = crop to valid pixels only
    )
    
    return StereoCalibrationData(
        K1=camera_matrix_left,
        K2=camera_matrix_right,
        D1=dist_coeffs_left.flatten(),
        D2=dist_coeffs_right.flatten(),
        R=R,
        T=T.flatten(),
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        image_size=image_size,
        rms_error=rms_error,
        calibration_date=datetime.now().isoformat()
    )
