"""
Phase 3: Detection Validation
Validates and filters AI detection results
"""

class Phase03DetectionValidation:
    """Phase 3: Validate detections"""
    
    def __init__(self):
        print("✅ Phase 3: Detection Validation initialized")
    
    def validate_detections(self, inference_result):
        """Validate detection results"""
        try:
            print("✅ Phase 3: Validating detections...")
            
            if not inference_result.get('success', False):
                return {
                    'success': False,
                    'error': 'Invalid inference result'
                }
            
            detections = inference_result.get('detections', [])
            
            if len(detections) == 0:
                return {
                    'success': False,
                    'error': 'No detections to validate'
                }
            
            # Basic validation - filter by confidence threshold
            valid_detections = [
                d for d in detections 
                if d.get('confidence', 0) > 0.5
            ]
            
            print(f"   ✅ Phase 3: Validated {len(valid_detections)}/{len(detections)} detections")
            
            return {
                'success': True,
                'validated_detections': valid_detections,
                'original_count': len(detections),
                'valid_count': len(valid_detections)
            }
            
        except Exception as e:
            print(f"   ❌ Phase 3 error: {str(e)}")
            return {
                'success': False,
                'error': f'Validation failed: {str(e)}'
            }
