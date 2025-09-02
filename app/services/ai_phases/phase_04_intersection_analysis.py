"""
Phase 4: Intersection Analysis
Analyzes intersections between detected rebar elements
"""

class Phase04IntersectionAnalysis:
    """Phase 4: Analyze intersections"""
    
    def __init__(self):
        print("🔄 Phase 4: Intersection Analysis initialized")
    
    def analyze_intersections(self, validation_result):
        """Analyze intersections between rebar elements"""
        try:
            print("🔄 Phase 4: Analyzing intersections...")
            
            if not validation_result.get('success', False):
                return {
                    'success': False,
                    'error': 'Invalid validation result'
                }
            
            detections = validation_result.get('validated_detections', [])
            
            # Basic intersection analysis
            intersections = []
            for i, det1 in enumerate(detections):
                for j, det2 in enumerate(detections[i+1:], i+1):
                    # Simple intersection check based on bounding boxes
                    bbox1 = det1.get('bbox', [0, 0, 0, 0])
                    bbox2 = det2.get('bbox', [0, 0, 0, 0])
                    
                    if self._boxes_intersect(bbox1, bbox2):
                        intersections.append({
                            'detection_1': i,
                            'detection_2': j,
                            'type': f"{det1.get('class_name', '')}_x_{det2.get('class_name', '')}"
                        })
            
            print(f"   ✅ Phase 4: Found {len(intersections)} intersections")
            
            return {
                'success': True,
                'intersections': intersections,
                'intersection_count': len(intersections)
            }
            
        except Exception as e:
            print(f"   ❌ Phase 4 error: {str(e)}")
            return {
                'success': False,
                'error': f'Intersection analysis failed: {str(e)}'
            }
    
    def _boxes_intersect(self, bbox1, bbox2):
        """Check if two bounding boxes intersect"""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
        except:
            return False
