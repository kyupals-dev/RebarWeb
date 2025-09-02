"""
AI Phases Package - Rebar Detection Pipeline
Complete modular processing pipeline for AI rebar analysis
"""

# Import all phase classes
try:
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
    
    # Make all phases available
    __all__ = [
        'Phase01ImagePreparation',
        'Phase02Detectron2Inference', 
        'Phase03DetectionValidation',
        'Phase04IntersectionAnalysis',
        'Phase05PolygonGeneration',
        'Phase06DimensionCalculation',
        'Phase07CementMixtureCalculation',
        'Phase08VisualizationCreation',
        'Phase09AutoSaveProcess',
        'Phase10ResponseFormatting'
    ]
    
    print("✅ AI Phases package loaded successfully")
    PHASES_IMPORT_SUCCESS = True
    
except ImportError as e:
    print(f"⚠️  Error importing AI phases: {e}")
    __all__ = []
    PHASES_IMPORT_SUCCESS = False
