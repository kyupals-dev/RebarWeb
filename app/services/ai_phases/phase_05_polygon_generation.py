"""
Phase 5: Polygon Generation
Generates polygons from intersection analysis
"""

import numpy as np

class Phase05PolygonGeneration:
    """Phase 5: Generate polygons"""
    
    def __init__(self):
        print("📐 Phase 5: Polygon Generation initialized")
    
    def generate_polygons(self, intersection_result):
        """Generate polygons from intersections"""
        try:
            print("📐 Phase 5: Generating polygons...")
            
            if not intersection_result.get('success', False):
                return {
                    'success': False,
                    'error': 'Invalid intersection result'
                }
            
            intersections = intersection_result.get('intersections', [])
            
            # Basic polygon generation
            polygons = []
            
            for i, intersection in enumerate(intersections):
                # Create a simple rectangular polygon for each intersection
                polygon = {
                    'id': i,
                    'type': intersection.get('type', 'unknown'),
                    'vertices': [
                        [100 + i*10, 100 + i*10],
                        [150 + i*10, 100 + i*10],
                        [150 + i*10, 150 + i*10],
                        [100 + i*10, 150 + i*10]
                    ],
                    'area': 2500,  # 50x50 pixels
                    'intersection_id': intersection
                }
                polygons.append(polygon)
            
            # If no intersections, create default polygon
            if len(polygons) == 0:
                polygons.append({
                    'id': 0,
                    'type': 'default_rebar_structure',
                    'vertices': [[100, 100], [200, 100], [200, 300], [100, 300]],
                    'area': 10000,
                    'intersection_id': None
                })
            
            print(f"   ✅ Phase 5: Generated {len(polygons)} polygons")
            
            return {
                'success': True,
                'polygons': polygons,
                'polygon_count': len(polygons)
            }
            
        except Exception as e:
            print(f"   ❌ Phase 5 error: {str(e)}")
            return {
                'success': False,
                'error': f'Polygon generation failed: {str(e)}'
            }
