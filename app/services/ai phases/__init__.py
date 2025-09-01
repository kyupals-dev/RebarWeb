"""
AI Phases Package
Modular AI processing phases for rebar detection and analysis
"""

# Import base phase
from .base_phase import BasePhase, PhaseError, PhaseValidationError

# Import all phases
from .phase_01_image_preparation import Phase01ImagePreparation
from .phase_02_detectron2_inference import Phase02Detectron2Inference
from .phase_03_detection_validation import Phase03DetectionValidation
from .phase_04_intersection_analysis import Phase04IntersectionAnalysis
from .phase_05_polygon_generation import Phase05PolygonGeneration
from .phase_06_dimension_calculation import Phase06DimensionCalculation
from .phase_07_cement_mixture_calculation import Phase07CementMixtureCalculation
from .phase_08_visualization_creation import Phase08VisualizationCreation
from .phase_09_auto_save_process import Phase09AutoSaveProcess
from .phase_10_response_formatting import Phase10ResponseFormatting

# Import utilities
from .utils import *

__all__ = [
    # Base classes
    'BasePhase',
    'PhaseError', 
    'PhaseValidationError',
    
    # Phase classes
    'Phase01ImagePreparation',
    'Phase02Detectron2Inference',
    'Phase03DetectionValidation',
    'Phase04IntersectionAnalysis',
    'Phase05PolygonGeneration',
    'Phase06DimensionCalculation',
    'Phase07CementMixtureCalculation',
    'Phase08VisualizationCreation',
    'Phase09AutoSaveProcess',
    'Phase10ResponseFormatting',
    
    # Utilities (imported from utils)
    'PHASE_NAMES',
    'CLASS_NAMES',
    'DEFAULT_COLORS',
    'DEFAULT_THRESHOLDS',
    'safe_get',
    'ensure_directory',
    'validate_image_path',
    'validate_detection_data'
]

# Phase mapping for easy access
PHASE_MAP = {
    1: Phase01ImagePreparation,
    2: Phase02Detectron2Inference, 
    3: Phase03DetectionValidation,
    4: Phase04IntersectionAnalysis,
    5: Phase05PolygonGeneration,
    6: Phase06DimensionCalculation,
    7: Phase07CementMixtureCalculation,
    8: Phase08VisualizationCreation,
    9: Phase09AutoSaveProcess,
    10: Phase10ResponseFormatting
}

def get_phase_class(phase_number):
    """Get phase class by number"""
    return PHASE_MAP.get(phase_number)

def get_all_phases():
    """Get list of all phase classes in order"""
    return [PHASE_MAP[i] for i in range(1, 11)]

def create_phase_instance(phase_number):
    """Create instance of specific phase"""
    phase_class = get_phase_class(phase_number)
    if phase_class:
        return phase_class()
    return None

def create_all_phase_instances():
    """Create instances of all phases"""
    return [phase_class() for phase_class in get_all_phases()]

# Version info
__version__ = '1.0.0'
__author__ = 'Rebar Vista Team'
__description__ = 'Modular AI phases for rebar detection and analysis'
