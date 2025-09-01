"""
Updated AI Service - Phase Coordinator
Coordinates all 10 phases of AI analysis in sequence
"""

import os
from datetime import datetime
import traceback

# Import all phases
from .ai_phases.phase_01_image_preparation import Phase01ImagePreparation
from .ai_phases.phase_02_detectron2_inference import Phase02Detectron2Inference
from .ai_phases.phase_03_detection_validation import Phase03DetectionValidation
from .ai_phases.phase_04_intersection_analysis import Phase04IntersectionAnalysis
from .ai_phases.phase_05_polygon_generation import Phase05PolygonGeneration
from .ai_phases.phase_06_dimension_calculation import Phase06DimensionCalculation
from .ai_phases.phase_07_cement_mixture_calculation import Phase07CementMixtureCalculation
from .ai_phases.phase_08_visualization_creation import Phase08VisualizationCreation
from .ai_phases.phase_09_auto_save_process import Phase09AutoSaveProcess
from .ai_phases.phase_10_response_formatting import Phase10ResponseFormatting

from .ai_phases.base_phase import PhaseError

class AIService:
    """AI Service - Coordinates all phases of rebar analysis"""
    
    def __init__(self):
        print("🤖 Initializing Modular AI Service with 10 Phases...")
        
        # Initialize all phases in order
        self.phases = [
            Phase01ImagePreparation(),      # Phase 1: Image Preparation
            Phase02Detectron2Inference(),   # Phase 2: Detectron2 Inference
            Phase03DetectionValidation(),   # Phase 3: Detection Validation
            Phase04IntersectionAnalysis(),  # Phase 4: Intersection Analysis
            Phase05PolygonGeneration(),     # Phase 5: Polygon Generation
            Phase06DimensionCalculation(),  # Phase 6: Dimension Calculation
            Phase07CementMixtureCalculation(),  # Phase 7: Cement Mixture Calculation
            Phase08VisualizationCreation(),     # Phase 8: Visualization Creation
            Phase09AutoSaveProcess(),           # Phase 9: Auto-Save Process
            Phase10ResponseFormatting()         # Phase 10: Response Formatting
        ]
        
        # Track overall status
        self.model_loaded = self.phases[1].model_loaded  # From Detectron2 phase
        self.total_phases = len(self.phases)
        
        print(f"✅ AI Service initialized with {self.total_phases} phases")
        print(f"   Model loaded: {'✅' if self.model_loaded else '❌'}")
        
        # Log all phases
        for i, phase in enumerate(self.phases, 1):
            print(f"   Phase {i}: {phase.phase_name}")
    
    def analyze_image(self, image_path):
        """
        Main analysis method - executes all phases in sequence
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Complete analysis results or error information
        """
        print("🚀 Starting complete AI analysis pipeline...")
        
        # Initialize data with image path
        data = {
            'image_path': image_path,
            'analysis_start_time': datetime.now(),
            'phase_results': {},
            'phase_timings': {}
        }
        
        try:
            # Execute all phases in sequence
            for i, phase in enumerate(self.phases, 1):
                phase_start_time = datetime.now()
                
                try:
                    print(f"\n📋 Phase {i}/{self.total_phases}: {phase.phase_name}")
                    
                    # Execute phase with timing
                    data = phase.execute_with_timing(data)
                    
                    # Store phase results
                    data['phase_results'][f'phase_{i:02d}'] = {
                        'name': phase.phase_name,
                        'success': True,
                        'timing': phase.get_timing_info()
                    }
                    
                    # Check for early termination signals
                    if data.get('should_stop_processing', False):
                        print(f"⚠️  Processing stopped at Phase {i}: {phase.phase_name}")
                        return self._create_early_termination_response(data, i)
                    
                    print(f"✅ Phase {i} completed successfully")
                    
                except Exception as phase_error:
                    print(f"❌ Phase {i} failed: {str(phase_error)}")
                    
                    # Store phase failure
                    data['phase_results'][f'phase_{i:02d}'] = {
                        'name': phase.phase_name,
                        'success': False,
                        'error': str(phase_error),
                        'timing': phase.get_timing_info() if hasattr(phase, 'get_timing_info') else None
                    }
                    
                    # Return error response
                    return self._create_error_response(data, i, phase_error)
            
            # All phases completed successfully
            analysis_end_time = datetime.now()
            total_time = (analysis_end_time - data['analysis_start_time']).total_seconds()
            
            print(f"\n🎉 Complete AI analysis pipeline finished!")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Phases completed: {self.total_phases}/{self.total_phases}")
            
            # Return the web response from Phase 10
            web_response = data.get('web_response', {})
            web_response['processing_time_total'] = f"{total_time:.2f}s"
            web_response['phases_completed'] = self.total_phases
            
            return web_response
            
        except Exception as e:
            print(f"💥 Critical error in AI analysis pipeline: {str(e)}")
            traceback.print_exc()
            return self._create_critical_error_response(data, e)
    
    def _create_early_termination_response(self, data, stopped_at_phase):
        """Create response for early termination (e.g., no detections)"""
        # Check the specific reason for stopping
        if data.get('validation_error') == 'no_detections':
            return {
                'success': False,
                'error': 'no_rebar_detected',
                'message': 'No rebar structures detected in the image',
                'phases_completed': stopped_at_phase,
                'total_phases': self.total_phases,
                'processing_details': data.get('phase_results', {})
            }
        
        elif data.get('validation_error') == 'missing_required_classes':
            return {
                'success': False,
                'error': 'insufficient_rebar_classes',
                'message': data.get('validation_message', 'Missing required rebar classes'),
                'phases_completed': stopped_at_phase,
                'total_phases': self.total_phases,
                'processing_details': data.get('phase_results', {})
            }
        
        elif data.get('intersection_error'):
            return {
                'success': False,
                'error': 'intersection_analysis_failed',
                'message': data.get('intersection_message', 'Failed to analyze rebar intersections'),
                'phases_completed': stopped_at_phase,
                'total_phases': self.total_phases,
                'processing_details': data.get('phase_results', {})
            }
        
        else:
            return {
                'success': False,
                'error': 'analysis_incomplete',
                'message': 'Analysis stopped due to insufficient data',
                'phases_completed': stopped_at_phase,
                'total_phases': self.total_phases,
                'processing_details': data.get('phase_results', {})
            }
    
    def _create_error_response(self, data, failed_phase, error):
        """Create response for phase failure"""
        return {
            'success': False,
            'error': 'phase_execution_failed',
            'message': f'Phase {failed_phase} failed: {str(error)}',
            'failed_phase': failed_phase,
            'failed_phase_name': self.phases[failed_phase - 1].phase_name,
            'phases_completed': failed_phase - 1,
            'total_phases': self.total_phases,
            'processing_details': data.get('phase_results', {}),
            'original_error': str(error)
        }
    
    def _create_critical_error_response(self, data, error):
        """Create response for critical system errors"""
        return {
            'success': False,
            'error': 'critical_system_error',
            'message': f'Critical system error: {str(error)}',
            'phases_completed': len(data.get('phase_results', {})),
            'total_phases': self.total_phases,
            'processing_details': data.get('phase_results', {}),
            'original_error': str(error)
        }
    
    def get_model_status(self):
        """Get current model status"""
        detectron2_phase = self.phases[1]  # Phase 2 is Detectron2 inference
        
        return {
            'service_type': 'modular_phase_based',
            'total_phases': self.total_phases,
            'detectron2_available': detectron2_phase.model_loaded,
            'model_loaded': self.model_loaded,
            'model_path': getattr(detectron2_phase, 'model_path', None),
            'model_exists': os.path.exists(getattr(detectron2_phase, 'model_path', '')) if getattr(detectron2_phase, 'model_path', None) else False,
            'num_classes': getattr(detectron2_phase, 'num_classes', 3),
            'class_names': getattr(detectron2_phase, 'class_names', []),
            'threshold': getattr(detectron2_phase, 'detection_threshold', 0.2),
            'phases_info': [
                {
                    'number': i + 1,
                    'name': phase.phase_name,
                    'class': phase.__class__.__name__
                }
                for i, phase in enumerate(self.phases)
            ]
        }
    
    def test_model(self, test_image_path=None):
        """Test the complete analysis pipeline with a sample image"""
        try:
            if not test_image_path:
                from app.utils.config import config
                
                # Use a recent captured image for testing
                if os.path.exists(config.UPLOAD_FOLDER):
                    images = [f for f in os.listdir(config.UPLOAD_FOLDER) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        test_image_path = os.path.join(config.UPLOAD_FOLDER, images[-1])
                    else:
                        return {
                            'success': False,
                            'error': 'No test images available'
                        }
                else:
                    return {
                        'success': False,
                        'error': 'Upload folder not found'
                    }
            
            print(f"🧪 Testing complete AI pipeline with: {test_image_path}")
            
            # Run complete analysis
            result = self.analyze_image(test_image_path)
            
            if result.get('success'):
                print("✅ Complete pipeline test successful!")
                return {
                    'success': True,
                    'test_image': test_image_path,
                    'phases_completed': result.get('phases_completed', 0),
                    'total_phases': self.total_phases,
                    'analysis_result_summary': {
                        'dimensions': result.get('dimensions', {}).get('display', 'N/A'),
                        'detections': result.get('detections', {}).get('count', 0),
                        'cement_mixture': result.get('cement_mixture', {}).get('ratio', 'N/A')
                    }
                }
            else:
                print(f"❌ Pipeline test failed: {result.get('message', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"❌ Pipeline test error: {str(e)}")
            return {
                'success': False,
                'error': f'Test failed: {str(e)}'
            }
    
    def get_phase_info(self, phase_number=None):
        """Get information about specific phase or all phases"""
        if phase_number is not None:
            if 1 <= phase_number <= self.total_phases:
                phase = self.phases[phase_number - 1]
                return {
                    'number': phase_number,
                    'name': phase.phase_name,
                    'class': phase.__class__.__name__,
                    'module': phase.__class__.__module__
                }
            else:
                return None
        else:
            return [
                {
                    'number': i + 1,
                    'name': phase.phase_name,
                    'class': phase.__class__.__name__,
                    'module': phase.__class__.__module__
                }
                for i, phase in enumerate(self.phases)
            ]
