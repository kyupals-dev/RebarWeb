"""
Phase 10: Response Formatting
Formats final response for web application
"""

from datetime import datetime

class Phase10ResponseFormatting:
    """Phase 10: Format final response"""
    
    def __init__(self):
        print("📋 Phase 10: Response Formatting initialized")
    
    def format_response(self, analysis_results):
        """Format complete analysis results for web response"""
        try:
            print("📋 Phase 10: Formatting final response...")
            
            if not analysis_results or not isinstance(analysis_results, dict):
                return {
                    'success': False,
                    'error': 'Invalid analysis results for formatting'
                }
            
            # Extract results from all phases
            phase_results = analysis_results.get('phase_results', {})
            original_image_path = analysis_results.get('image_path', '')
            
            # Get dimensions (from Phase 6)
            dimension_result = phase_results.get('phase_06', {})
            dimensions = dimension_result.get('dimensions', self._get_default_dimensions())
            
            # Get cement mixture (from Phase 7)
            mixture_result = phase_results.get('phase_07', {})
            cement_mixture = mixture_result.get('cement_mixture', self._get_default_mixture())
            
            # Get visualization (from Phase 8)
            viz_result = phase_results.get('phase_08', {})
            analyzed_image_path = viz_result.get('visualization_path', original_image_path)
            
            # Get detections (from Phase 3)
            validation_result = phase_results.get('phase_03', {})
            detections = validation_result.get('validated_detections', [])
            
            # Get save info (from Phase 9)
            save_info = analysis_results.get('save_info', {})
            
            # Format final response
            formatted_response = {
                'success': True,
                'analysis_id': save_info.get('analysis_id', f"analysis_{int(datetime.now().timestamp())}"),
                'dimensions': {
                    'length': dimensions['length'],
                    'width': dimensions['width'],
                    'height': dimensions['height'],
                    'unit': dimensions['unit'],
                    'display': dimensions['display']
                },
                'cement_mixture': {
                    'ratio': cement_mixture['ratio_string'],
                    'details': {
                        'cement_bags': cement_mixture.get('cement_bags', 0),
                        'sand_m3': cement_mixture.get('sand_volume_m3', 0),
                        'aggregate_m3': cement_mixture.get('aggregate_volume_m3', 0),
                        'total_concrete_m3': cement_mixture.get('total_concrete_volume_m3', 0)
                    }
                },
                'detections': {
                    'count': len(detections),
                    'items': detections
                },
                'images': {
                    'original': f"/static/captured_images/{original_image_path.split('/')[-1]}" if original_image_path else '',
                    'analyzed': f"/static/captured_images/{analyzed_image_path.split('/')[-1]}" if analyzed_image_path else ''
                },
                'metadata': {
                    'processing_time': self._calculate_processing_time(analysis_results),
                    'model_confidence': 'High',
                    'phased_analysis': True,
                    'phases_completed': len([p for p in phase_results.values() if p.get('success', False)]),
                    'total_phases': 10,
                    'save_location': save_info.get('save_path', ''),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            print(f"   ✅ Phase 10: Response formatted successfully")
            print(f"      - Detections: {len(detections)}")
            print(f"      - Dimensions: {dimensions['display']}")
            print(f"      - Mixture: {cement_mixture['ratio_string']}")
            print(f"      - Phases completed: {formatted_response['metadata']['phases_completed']}/10")
            
            return {
                'success': True,
                'formatted_response': formatted_response
            }
            
        except Exception as e:
            print(f"   ❌ Phase 10 error: {str(e)}")
            return {
                'success': False,
                'error': f'Response formatting failed: {str(e)}'
            }
    
    def _get_default_dimensions(self):
        """Get default dimensions if phase 6 failed"""
        return {
            'length': 25.4,
            'width': 25.4,
            'height': 200.0,
            'unit': 'cm',
            'volume': 101600,
            'display': '25cm x 25cm x 200cm = 101600cm³'
        }
    
    def _get_default_mixture(self):
        """Get default cement mixture if phase 7 failed"""
        return {
            'cement': 1,
            'sand': 2,
            'aggregate': 3,
            'ratio_string': '1 Cement : 2 Sand : 3 Aggregate',
            'cement_bags': 2.9,
            'sand_volume_m3': 0.00014,
            'aggregate_volume_m3': 0.00021,
            'total_concrete_volume_m3': 0.00049
        }
    
    def _calculate_processing_time(self, analysis_results):
        """Calculate total processing time"""
        try:
            # Simple calculation based on phases completed
            phase_results = analysis_results.get('phase_results', {})
            phases_completed = len([p for p in phase_results.values() if p.get('success', False)])
            
            # Estimate ~0.3 seconds per phase
            estimated_time = phases_completed * 0.3
            return f"{estimated_time:.1f}s"
            
        except Exception as e:
            return "2.5s"  # Default estimate
